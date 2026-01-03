"""
Description:
The output of anonymizer.py is a dataframe anonymized both in column names and data values, 
and a mapping dictionary for original to anonymized column names. The LLM will ostensibly return 
a function or set of functions buit on the anonymized data. The deanonymizer_helper.py provides 
two functions. (a) rename the original dataframe columns to the anonymized names, and (b) take 
anonymized function code and replace the anonymized column names with the original column names.
"""


def rename_to_anonymized(df, mapping):
    """
    Rename original column names to anonymized column names.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with original column names
    mapping : dict
        {original_name -> anonymized_name}

    Returns
    -------
    pandas.DataFrame
    """
    return df.rename(columns=mapping)


def rename_to_original(df, mapping):
    """
    Rename anonymized column names back to original names.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with anonymized column names
    mapping : dict
        {original_name -> anonymized_name}

    Returns
    -------
    pandas.DataFrame
    """
    reverse = {v: k for k, v in mapping.items()}
    return df.rename(columns=reverse)