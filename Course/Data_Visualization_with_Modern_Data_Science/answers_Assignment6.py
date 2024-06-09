import pandas as pd
import numpy as np

def retrieve_candidates_and_parties_from_regional_legislators_taipei_xlsx() -> pd.core.frame.DataFrame:
  """
  >>> candidates_and_parties = retrieve_candidates_and_parties_from_regional_legislators_taipei_xlsx()
  >>> candidates_and_parties.shape
  (53, 3)
  """
  ### BEGIN SOLUTION
  number = [8,5,7,5,10,5,5,8]
  list = []
  list2 = []
  list3 = []
  for i in range(8):
    regional_legislators_taipei_xlsx = pd.read_excel('regional_legislators_taipei.xlsx', sheet_name=i)
    candidates_info = regional_legislators_taipei_xlsx.iloc[1,3:3+number[i]]
    candidates_info = candidates_info.str.split(expand=True)
    list.extend(candidates_info.values[:,1])
  for i in range(8):
    regional_legislators_taipei_xlsx = pd.read_excel('regional_legislators_taipei.xlsx', sheet_name=i)
    candidates_info = regional_legislators_taipei_xlsx.iloc[1,3:3+number[i]]
    candidates_info = candidates_info.str.split(expand=True)
    list2.extend(candidates_info.values[:,2])
  for i in range(8):
    for j in range(number[i]):
      list3.append(f'臺北市第{i+1}選舉區')
  result = np.array([list3,list2,list])
  return(pd.DataFrame(result.T))
  ### END SOLUTION

def retrieve_votes_received_from_regional_legislators_taipei_xlsx() -> pd.core.frame.DataFrame:
  """
  >>> votes_received = retrieve_votes_received_from_regional_legislators_taipei_xlsx()
  >>> votes_received.shape
  (53, 4)
  """
  ### BEGIN SOLUTION
  a = retrieve_candidates_and_parties_from_regional_legislators_taipei_xlsx()
  number = [8,5,7,5,10,5,5,8]
  votes = []
  for i in range(8):
    regional_legislators_taipei_xlsx = pd.read_excel('regional_legislators_taipei.xlsx', sheet_name=i)
    count = regional_legislators_taipei_xlsx.iloc[4,3:3+number[i]]
    votes.extend(count.values)
  result = np.array(a)
  new_result = np.column_stack((result,votes))
  return(pd.DataFrame(new_result))
  ### END SOLUTION




def retrieve_elected_from_regional_legislators_taipei_xlsx() -> pd.core.frame.DataFrame:
  """
  >>> elected = retrieve_elected_from_regional_legislators_taipei_xlsx()
  >>> elected.shape
  (8, 4)
  """
  ### BEGIN SOLUTION
  a = retrieve_votes_received_from_regional_legislators_taipei_xlsx()
  sizes = [8,5,7,5,10,5,5,8]
  count = 0
  final = np.empty(8)
  data = np.array(a)
  
  sub_arrays = []
  start_idx = 0
  for size in sizes:
      sub_arrays.append(a[start_idx:start_idx + size])
      start_idx += size

  print(sub_arrays[0][])
  
  for i in range(8):
    if count == 0:
      winner = sub_arrays[i][count]
    elif count != 0 and sub_arrays[i][count,3] > winner[3]:
      winner = sub_arrays[i][count]
    count += 1
    print(winner)
    #print(data[i])
  

  ### END SOLUTION