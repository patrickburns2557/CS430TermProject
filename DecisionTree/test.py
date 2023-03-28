import pandas as pd

name_dict = {
            'Name': ['a','b','c','d'],
            'Score': [90,80,95,20],
            'test': ["example1", "example2", "example3", "example4"]
          }

df = pd.DataFrame(name_dict)

split = df["Name", "test"]

print (df)