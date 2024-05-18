import sqlite3
import pandas as pd

def import_presidents_csv(csv_file_path: str) -> pd.core.frame.DataFrame:
  """
  >>> presidents_csv = import_presidents_csv("presidents.csv")
  >>> presidents_csv.shape
  (53385, 7)
  """
  ### BEGIN SOLUTION
  csv_file_path = "presidents.csv"
  presidents_csv = pd.read_csv(csv_file_path)
  return presidents_csv
  ### END SOLUTION

def return_presidents_csv_shape(df: pd.core.frame.DataFrame, dim_name: str) -> int:
  """
  >>> presidents_csv = import_presidents_csv("presidents.csv")
  >>> return_presidents_csv_shape(presidents_csv, "rows")
  53385
  >>> return_presidents_csv_shape(presidents_csv, "columns")
  7
  """
  ### BEGIN SOLUTION
  if dim_name.lower() == "rows":
    return df.shape[0] 
  elif dim_name.lower() == "columns":
    return df.shape[1] 
  ### END SOLUTION  

def return_presidents_csv_head_tail(df: pd.core.frame.DataFrame, head_or_tail: str) -> pd.core.frame.DataFrame:
  """
  >>> presidents_csv = import_presidents_csv("presidents.csv")
  >>> return_presidents_csv_head_tail(presidents_csv, "head").shape
  (5, 7)
  >>> return_presidents_csv_head_tail(presidents_csv, "tail").shape
  (5, 7)
  """
  ### BEGIN SOLUTION
  if head_or_tail.lower() == "head":
    return df.head(5) 
  elif head_or_tail.lower() == "tail":
    return df.tail(5) 
  ### END SOLUTION

def return_presidents_csv_columns(df: pd.core.frame.DataFrame):
  """
  >>> presidents_csv = import_presidents_csv("presidents.csv")
  >>> "candidate_id" in return_presidents_csv_columns(presidents_csv)
  True
  >>> "votes" in return_presidents_csv_columns(presidents_csv)
  True
  """
  ### BEGIN SOLUTION
  return df.columns
  ### END SOLUTION

def summarize_presidents_csv(df: pd.core.frame.DataFrame, column_name: str):
  """
  >>> presidents_csv = import_presidents_csv("presidents.csv")
  >>> summarize_presidents_csv(presidents_csv, "number")
  (1, 2, 3)
  >>> summarize_presidents_csv(presidents_csv, "candidate_id")
  (329, 330, 331)
  >>> summarize_presidents_csv(presidents_csv, "votes")
  {1: 3690466, 2: 5586019, 3: 4671021}
  """
  ### BEGIN SOLUTION
  column_data = df[column_name]

  if column_name == "number" or column_name == "candidate_id":
      return tuple(sorted(column_data.unique()))
  elif column_name == "votes":
      sum = df.groupby('number')['votes'].sum()
      return sum
    
  ### END SOLUTION

def create_sqlite3_connection():
  """
  >>> conn = create_sqlite3_connection()
  >>> type(conn)
  <class 'sqlite3.Connection'>
  """
  ### BEGIN SOLUTION
  conn = sqlite3.connect("taiwan_election_2024.db")
  return conn
  ### END SOLUTION

def import_presidents_from_sqlite_db(connection: sqlite3.Connection) -> pd.core.frame.DataFrame:
  """
  >>> conn = create_sqlite3_connection()
  >>> presidents = import_presidents_from_sqlite_db(conn)
  >>> presidents.shape
  (53385, 7)
  """
  ### BEGIN SOLUTION
  conn = sqlite3.connect("taiwan_election_2024.db")
  sql_query = """
  SELECT *
    FROM presidents;
  """
  presidents = pd.read_sql_query(sql_query, conn)
  return presidents
  ### END SOLUTION

def import_table_from_sqlite_db(connection: sqlite3.Connection, table_name: str) -> pd.core.frame.DataFrame:
  """
  >>> conn = create_sqlite3_connection()
  >>> election_types = import_table_from_sqlite_db(conn, "election_types")
  >>> election_types.shape
  (5, 2)
  >>> parties = import_table_from_sqlite_db(conn, "parties")
  >>> parties.shape
  (35, 2)
  """
  ### BEGIN SOLUTION
  conn = sqlite3.connect("taiwan_election_2024.db")
  list = ["election_types","parties","candidates","presidents"]
  if table_name == list[0]:
    sql_query = """
    SELECT *
      FROM election_types;
    """
    a = pd.read_sql_query(sql_query, conn)
    return a

  elif table_name == list[1]:
    sql_query = """
    SELECT *
      FROM parties;
    """
    b = pd.read_sql_query(sql_query, conn)
    return b

  elif table_name == list[2]:
    sql_query = """
    SELECT *
      FROM candidates;
    """
    c = pd.read_sql_query(sql_query, conn)
    return c

  elif table_name == list[3]:
    sql_query = """
    SELECT *
      FROM presidents;
    """
    d = pd.read_sql_query(sql_query, conn)
    return d
    
  ### END SOLUTION

def extract_table_columns_from_sqlite_db(connection: sqlite3.Connection, table_name: str):
  """
  >>> conn = create_sqlite3_connection()
  >>> "id" in extract_table_columns_from_sqlite_db(conn, "election_types")
  True
  >>> "election_type" in extract_table_columns_from_sqlite_db(conn, "election_types")
  True
  >>> "id" in extract_table_columns_from_sqlite_db(conn, "parties")
  True
  >>> "name" in extract_table_columns_from_sqlite_db(conn, "parties")
  True
  """
  ### BEGIN SOLUTION
  conn = sqlite3.connect("taiwan_election_2024.db")
  list = ["election_types","parties","candidates","presidents"]
  if table_name == list[0]:
    sql_query = """
    SELECT *
      FROM PRAGMA_TABLE_INFO('election_types');
    """
    a = pd.read_sql_query(sql_query, conn)
    name = []
    name.append(a["name"][0])
    name.append(a["name"][1])
    return name

  elif table_name == list[1]:
    sql_query = """
    SELECT name
      FROM PRAGMA_TABLE_INFO('parties');
    """
    b = pd.read_sql_query(sql_query, conn)
    name = []
    name.append(b["name"][0])
    name.append(b["name"][1])
    return name

  elif table_name == list[2]:
    sql_query = """
    SELECT name
      FROM PRAGMA_TABLE_INFO('candidates');
    """
    c = pd.read_sql_query(sql_query, conn)
    name = []
    name.append(c["name"][0])
    name.append(c["name"][1])
    name.append(c["name"][2])
    name.append(c["name"][3])
    return name
  ### END SOLUTION

def show_table_shapes_from_sqlite_db(connection: sqlite3.Connection) -> dict:
  """
  >>> conn = create_sqlite3_connection()
  >>> len(show_table_shapes_from_sqlite_db(conn))
  10
  >>> "election_types" in show_table_shapes_from_sqlite_db(conn).keys()
  True
  >>> "parties" in show_table_shapes_from_sqlite_db(conn).keys()
  True
  >>> show_table_shapes_from_sqlite_db(conn)["election_types"]
  (5, 2)
  >>> show_table_shapes_from_sqlite_db(conn)["parties"]
  (35, 2)
  """
  ### BEGIN SOLUTION
  conn = sqlite3.connect("taiwan_election_2024.db")
  sql_query = """
  SELECT name
    FROM sqlite_master
    limit 10;
  """
  a = pd.read_sql_query(sql_query, conn)
  dict = {}
 
  for i in range(10):
    dict[a["name"][i]] = pd.read_sql(f'select * from {a["name"][i]}' , conn).shape 

  return dict
  ### END SOLUTION