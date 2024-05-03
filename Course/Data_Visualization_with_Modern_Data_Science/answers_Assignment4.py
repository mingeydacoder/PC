from re import X

#1
def check_data_type(x) -> str:
  """
  >>> check_data_type(1)
  'int'
  >>> check_data_type(1.0)
  'float'
  >>> check_data_type(False)
  'bool'
  >>> check_data_type(True)
  'bool'
  >>> check_data_type('5566')
  'str'
  >>> check_data_type(None)
  'NoneType'
  """
  ### BEGIN SOLUTION
  return type(x).__name__
  ### END SOLUTION

#2
def check_data_structure(x) -> str:
  """
  >>> check_data_structure([2, 3, 5, 7, 11])
  'list'
  >>> check_data_structure((2, 3, 5, 7, 11))
  'tuple'
  >>> check_data_structure({'0': 2, '1': 3, '2': 5, '3': 7, '4': 11})
  'dict'
  >>> check_data_structure({2, 3, 5, 7, 11})
  'set'
  """
  ### BEGIN SOLUTION
  if isinstance(x, list):
    return "list"
  elif isinstance(x, dict):
    return "dict"
  elif isinstance(x, tuple):
    return "tuple"
  elif isinstance(x, set):
    return "set"
  elif isinstance(x, str):
    return "string"
  elif isinstance(x, int):
    return "integer"
  elif isinstance(x, float):
    return "float"
  else:
    return type(x).__name__
  ### END SOLUTION

#3
def retrieve_first_and_last_elements(x: list) -> tuple:
  """
  >>> retrieve_first_and_last_elements([2, 3, 5])
  (2, 5)
  >>> retrieve_first_and_last_elements([2, 3, 5, 7])
  (2, 7)
  >>> retrieve_first_and_last_elements(["Frieren", "Heiter", "Eisen", "Himmel"])
  ('Frieren', 'Himmel')
  """
  ### BEGIN SOLUTION
  if len(x) == 1:  
    return (x[0], x[0])
  else:
    return (x[0], x[-1])
  ### END SOLUTION

#4
def retrieve_middle_elements(x: list):
  """
  >>> retrieve_middle_elements([2, 3, 5])
  3
  >>> retrieve_middle_elements([2, 3, 5, 7])
  (3, 5)
  >>> retrieve_middle_elements([2, 3, 5, 7, 11])
  5
  >>> retrieve_middle_elements([2, 3, 5, 7, 11, 13])
  (5, 7)
  >>> retrieve_middle_elements(["Heiter", "Frieren", "Himmel", "Eisen"])
  ('Frieren', 'Himmel')
  """
  ### BEGIN SOLUTION
  if not x: 
    return None
  length = len(x)
  middle_index = length // 2
  if length % 2 == 0:  
    return (x[middle_index - 1], x[middle_index])
  else: 
    return (x[middle_index])
  ### END SOLUTION

#5
def median(x: list):
  """
  >>> median([2, 3, 5, 7, 11])
  5
  >>> median([2, 3, 5, 7, 11, 13])
  6.0
  >>> median([11, 13, 17, 2, 3, 5, 7])
  7
  >>> median([7, 11, 13, 17, 19, 2, 3, 5])
  9.0
  """
  ### BEGIN SOLUTION
  if not x:  
    return None
  sorted_numbers = sorted(x)  
  length = len(sorted_numbers)
  middle_index = length // 2

  if length % 2 == 0: 
    return (sorted_numbers[middle_index - 1] + sorted_numbers[middle_index]) / 2.0
  else: 
    return sorted_numbers[middle_index]
  ### END SOLUTION

#6
def collect_divisors(x: int) -> set:
  """
  >>> collect_divisors(1)
  {1}
  >>> collect_divisors(2)
  {1, 2}
  >>> collect_divisors(3)
  {1, 3}
  >>> collect_divisors(4)
  {1, 2, 4}
  >>> collect_divisors(5)
  {1, 5}
  """
  ### BEGIN SOLUTION
  divisors = set()
  for i in range(1, int(x**0.5) + 1):
      if x % i == 0:  
          divisors.add(i)
          divisors.add(x // i) 

  return (divisors)
  ### END SOLUTION

#7
def is_prime(x: int) -> bool:
  """
  >>> is_prime(1)
  False
  >>> is_prime(2)
  True
  >>> is_prime(3)
  True
  >>> is_prime(4)
  False
  >>> is_prime(5)
  True
  """
  ### BEGIN SOLUTION
  if x <= 1:
      return False  
  if x <= 3:
      return True  
  if x % 2 == 0 or x % 3 == 0:
      return False  
  i = 5
  while i * i <= x:
      if x % i == 0 or x % (i + 2) == 0:
          return False
      i += 6  
  return True
  ### END SOLUTION

#8
def list_first_n_prime_numbers(n: int) -> list:
  """
  >>> list_first_n_prime_numbers(5)
  [2, 3, 5, 7, 11]
  >>> list_first_n_prime_numbers(10)
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  >>> list_first_n_prime_numbers(30)
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
  """
  ### BEGIN SOLUTION
  primes = []
  candidate = 2  

  while len(primes) < n:
      if is_prime(candidate):
          primes.append(candidate)
      candidate += 1

  return primes
  ### END SOLUTION

#9
def swap_vowel_case(x: str) -> str:
  """
  >>> swap_vowel_case('a')
  'A'
  >>> swap_vowel_case('b')
  'b'
  >>> swap_vowel_case('c')
  'c'
  >>> swap_vowel_case('d')
  'd'
  >>> swap_vowel_case('e')
  'E'
  >>> swap_vowel_case('A')
  'a'
  >>> swap_vowel_case('B')
  'B'
  >>> swap_vowel_case('C')
  'C'
  >>> swap_vowel_case('D')
  'D'
  >>> swap_vowel_case('E')
  'e'
  """
  ### BEGIN SOLUTION
  vowels = 'aeiouAEIOU' 
  swapped = [] 

  for char in x:
      if char in vowels:
          if char.islower():
              swapped.append(char.upper())  
          else:
              swapped.append(char.lower())  
      else:
          swapped.append(char) 

  return ''.join(swapped) 
  ### END SOLUTION

#10
def swap_vowels_case_in_word(x: str) -> str:
  """
  >>> swap_vowels_case_in_word('Frieren')
  'FrIErEn'
  >>> swap_vowels_case_in_word('Himmel')
  'HImmEl'
  """
  ### BEGIN SOLUTION
  vowels = 'aeiouAEIOU' 
  swapped = [] 

  for char in x:
      if char in vowels:
          if char.islower():
              swapped.append(char.upper())  
          else:
              swapped.append(char.lower())  
      else:
          swapped.append(char) 

  return ''.join(swapped) 
  ### END SOLUTION