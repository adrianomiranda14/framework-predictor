



def lower_case(dataframe):
  """ Transforms all values in columns to lower case """
  for col in dataframe.columns:
    if dataframe[col].dtypes == 'object':
      dataframe[col] = dataframe[col].str.lower()
  return dataframe



def clean_columns(path):
  "This cleans column names of the csv passed through"
  df = pd.read_csv(path,  low_memory=False)
  df.columns = [x.lower() for x in df.columns]
  df.columns = [x.replace(" ","_") for x in df.columns]
  df.columns = [x.replace(",","") for x in df.columns]
  df.columns = [x.replace(")","") for x in df.columns]
  df.columns = [x.replace("(","") for x in df.columns]
  #df['job_role'] = df['job_role'].str.lower()
  return df