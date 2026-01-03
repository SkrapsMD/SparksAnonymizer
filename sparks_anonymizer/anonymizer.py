from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class AnonymizeConfig:
    seed: Optional[int] = 0
    # Column naming
    prefix: str = "col"
    # String behavior
    string_strategy: str = "random_like"   # {"random_like", "stable_pseudonym", "redact"}
    keep_string_length: bool = True
    keep_string_charset: bool = True       # preserve digit/alpha/punct pattern in random_like
    # Category behavior
    preserve_category_frequencies: bool = True
    # Numeric behavior
    numeric_strategy: str = "range"        # {"range", "normal_like"}
    clamp_to_observed: bool = True
    # Datetime behavior
    datetime_strategy: str = "range"       # {"range"}
    preserve_tz: bool = True
    # Misc
    preserve_missingness: bool = True


class SparksAnonymizer:
    """
    Structure-preserving synthetic data generator for restricted datasets.

    Returns:
      - anonymized_df: same shape, similar dtype schema
      - col_mapping: dict original_col -> anonymized_col
    """

    def __init__(self, config: Optional[AnonymizeConfig] = None):
        self.config = config or AnonymizeConfig()
        self.rng = np.random.default_rng(self.config.seed)

    # ----------------------------
    # Column name anonymization
    # ----------------------------
    def _dtype_to_base(self, dtype: Any) -> str:
        c = self.config.prefix

        if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            return f"{c}_str"
        if pd.api.types.is_categorical_dtype(dtype):
            return f"{c}_cat"
        if pd.api.types.is_bool_dtype(dtype):
            return f"{c}_bool"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return f"{c}_dt"
        if pd.api.types.is_integer_dtype(dtype):
            # dtype may be "int64" / "Int64" etc
            return f"{c}_{str(dtype)}"
        if pd.api.types.is_float_dtype(dtype):
            return f"{c}_{str(dtype)}"
        return f"{c}_{str(dtype)}"

    def anonymize_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        counts: Dict[str, int] = {}
        mapping: Dict[str, str] = {}

        for col in df.columns:
            base = self._dtype_to_base(df[col].dtype)
            counts[base] = counts.get(base, 0) + 1
            mapping[col] = f"{base}_{counts[base]}"

        return mapping

    # ----------------------------
    # String anonymization
    # ----------------------------
    def _stable_token(self, text: str, salt: str) -> str:
        # Deterministic token using sha256 (NOT reversible)
        h = hashlib.sha256((salt + "\x1f" + text).encode("utf-8")).hexdigest()
        return h[:12]  # short token

    def _random_like_string(self, s: str) -> str:
        # preserve length + (optionally) pattern of alpha/digit/punct
        if not s:
            return s

        out = []
        for ch in s:
            if self.config.keep_string_charset:
                if ch.isupper():
                    out.append(chr(self.rng.integers(ord("A"), ord("Z") + 1)))
                elif ch.islower():
                    out.append(chr(self.rng.integers(ord("a"), ord("z") + 1)))
                elif ch.isdigit():
                    out.append(str(int(self.rng.integers(0, 10))))
                else:
                    # keep punctuation/whitespace as-is
                    out.append(ch)
            else:
                # fully random letters/digits, keep punctuation
                if ch.isalnum():
                    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    out.append(alphabet[int(self.rng.integers(0, len(alphabet)))])
                else:
                    out.append(ch)

        return "".join(out)

    def _anonymize_string_series(self, s: pd.Series, col_salt: str) -> pd.Series:
        strat = self.config.string_strategy

        mask = s.isna()
        s_filled = s.astype("string")

        if strat == "redact":
            fake = pd.Series(["[REDACTED]"] * len(s_filled), index=s_filled.index, dtype="string")
        elif strat == "stable_pseudonym":
            # preserve uniqueness relationships + allow joins if same salt used
            def f(x: Any) -> Any:
                if pd.isna(x):
                    return x
                tok = self._stable_token(str(x), salt=col_salt)
                if self.config.keep_string_length:
                    # keep approximate length by padding/truncating token
                    target = len(str(x))
                    if target <= 0:
                        return ""
                    if target <= len(tok):
                        return tok[:target]
                    return (tok * math.ceil(target / len(tok)))[:target]
                return tok

            fake = s_filled.map(f).astype("string")
        else:  # "random_like"
            def f(x: Any) -> Any:
                if pd.isna(x):
                    return x
                x_str = str(x)
                if not self.config.keep_string_length:
                    # random length between 4 and 16
                    L = int(self.rng.integers(4, 17))
                    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    return "".join(alphabet[int(self.rng.integers(0, len(alphabet)))] for _ in range(L))
                return self._random_like_string(x_str)

            fake = s_filled.map(f).astype("string")

        if self.config.preserve_missingness:
            fake[mask] = pd.NA
        return fake

    # ----------------------------
    # Numeric anonymization
    # ----------------------------
    def _anonymize_int_series(self, s: pd.Series) -> pd.Series:
        mask = s.isna()
        # Work in numpy, but preserve pandas nullable dtype if present
        dtype = s.dtype

        # Edge cases: all NA
        if mask.all():
            return s.copy()

        s_non = s[~mask].astype("int64", errors="ignore")

        minv = int(np.nanmin(s_non))
        maxv = int(np.nanmax(s_non))

        if minv == maxv:
            # generate small noise around the constant, still integer
            span = 10
            minv2, maxv2 = minv - span, maxv + span
        else:
            minv2, maxv2 = minv, maxv

        n = (~mask).sum()
        vals = self.rng.integers(minv2, maxv2 + 1, size=n, dtype="int64")

        out = s.copy()
        out.loc[~mask] = vals

        # restore dtype (nullable Int64 stays nullable)
        try:
            out = out.astype(dtype)
        except Exception:
            pass

        return out

    def _anonymize_float_series(self, s: pd.Series) -> pd.Series:
        mask = s.isna()
        if mask.all():
            return s.copy()

        dtype = s.dtype
        s_non = s[~mask].astype("float64")

        minv = float(np.nanmin(s_non))
        maxv = float(np.nanmax(s_non))

        n = (~mask).sum()

        if self.config.numeric_strategy == "normal_like" and np.isfinite(minv) and np.isfinite(maxv):
            mu = float(np.nanmean(s_non))
            sigma = float(np.nanstd(s_non))
            sigma = sigma if sigma > 0 else (abs(mu) * 0.1 + 1.0)
            vals = self.rng.normal(mu, sigma, size=n)
            if self.config.clamp_to_observed:
                vals = np.clip(vals, minv, maxv)
        else:
            if not (np.isfinite(minv) and np.isfinite(maxv)) or minv == maxv:
                # fallback
                mu = float(np.nanmean(s_non))
                vals = self.rng.normal(mu, 1.0, size=n)
            else:
                vals = self.rng.uniform(minv, maxv, size=n)

        out = s.copy()
        out.loc[~mask] = vals

        try:
            out = out.astype(dtype)
        except Exception:
            out = out.astype("float64")

        return out

    # ----------------------------
    # Boolean anonymization
    # ----------------------------
    def _anonymize_bool_series(self, s: pd.Series) -> pd.Series:
        mask = s.isna()
        if mask.all():
            return s.copy()

        # preserve original True share if possible
        s_non = s[~mask]
        try:
            p_true = float(s_non.mean())  # works if boolean-ish
            p_true = min(max(p_true, 0.0), 1.0)
        except Exception:
            p_true = 0.5

        n = (~mask).sum()
        vals = self.rng.random(n) < p_true

        out = s.copy()
        out.loc[~mask] = vals

        # restore dtype if pandas nullable boolean
        try:
            out = out.astype(s.dtype)
        except Exception:
            out = out.astype("bool", errors="ignore")

        return out

    # ----------------------------
    # Datetime anonymization
    # ----------------------------
    def _anonymize_datetime_series(self, s: pd.Series) -> pd.Series:
        mask = s.isna()
        if mask.all():
            return s.copy()

        # Handle tz-aware separately
        tz = None
        if hasattr(s.dtype, "tz") and s.dtype.tz is not None:
            tz = s.dtype.tz

        s_non = s[~mask]

        # convert to int ns
        if tz is not None:
            s_non_ns = s_non.dt.tz_convert("UTC").view("int64")
        else:
            s_non_ns = s_non.view("int64")

        minv = int(np.min(s_non_ns))
        maxv = int(np.max(s_non_ns))

        n = (~mask).sum()
        if minv == maxv:
            # jitter within +/- 7 days
            span = int(pd.Timedelta(days=7).value)
            vals = self.rng.integers(minv - span, maxv + span + 1, size=n, dtype="int64")
        else:
            vals = self.rng.integers(minv, maxv + 1, size=n, dtype="int64")

        out = s.copy()
        ts = pd.to_datetime(vals, utc=(tz is not None))
        if tz is not None and self.config.preserve_tz:
            ts = ts.tz_convert(tz)
        elif tz is not None:
            # drop tz
            ts = ts.tz_convert(None)

        out.loc[~mask] = ts

        return out

    # ----------------------------
    # Categorical anonymization
    # ----------------------------
    def _anonymize_categorical_series(self, s: pd.Series, col_salt: str) -> pd.Series:
        mask = s.isna()
        if mask.all():
            return s.copy()

        cat = s.astype("category")
        cats = list(cat.cat.categories)

        # create fake labels for categories
        fake_labels = []
        for i, c in enumerate(cats, start=1):
            base = self._stable_token(str(c), salt=col_salt)
            fake_labels.append(f"cat_{i}_{base[:6]}")

        mapping = dict(zip(cats, fake_labels))

        if self.config.preserve_category_frequencies:
            # remap values directly; frequencies preserved exactly
            out = cat.map(mapping).astype("category")
        else:
            # sample categories uniformly for non-missing
            n = (~mask).sum()
            sampled = self.rng.choice(fake_labels, size=n, replace=True)
            out = pd.Series(pd.Categorical([None] * len(s), categories=fake_labels), index=s.index)
            out.loc[~mask] = sampled

        if self.config.preserve_missingness:
            out.loc[mask] = pd.NA

        return out

    # ----------------------------
    # Main entry point
    # ----------------------------
    def anonymize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        col_map = self.anonymize_columns(df)
        out = df.copy(deep=True)

        # Apply per-column dtype-aware anonymization
        for col in df.columns:
            series = df[col]
            salt = f"{col}|{self.config.seed}|{str(series.dtype)}"

            if pd.api.types.is_categorical_dtype(series.dtype):
                out[col] = self._anonymize_categorical_series(series, col_salt=salt)

            elif pd.api.types.is_datetime64_any_dtype(series.dtype):
                out[col] = self._anonymize_datetime_series(series)

            elif pd.api.types.is_bool_dtype(series.dtype):
                out[col] = self._anonymize_bool_series(series)

            elif pd.api.types.is_integer_dtype(series.dtype):
                out[col] = self._anonymize_int_series(series)

            elif pd.api.types.is_float_dtype(series.dtype):
                out[col] = self._anonymize_float_series(series)

            else:
                # object/string fallback
                out[col] = self._anonymize_string_series(series, col_salt=salt)

        # Rename columns last (keeps logic based on original names/dtypes)
        out = out.rename(columns=col_map)

        return out, col_map



