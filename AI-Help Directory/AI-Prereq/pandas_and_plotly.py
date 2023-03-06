import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

"""
Author: Rohan Singh
Date: November 16, 2022
Note: You will have to change some of the variable names and arguments for this code to work
"""

#This will make the figures display on the default browser
pio.renderers.default = 'browser'

def main():
  file_path = "/Path/to/csv/file/here"
  
  #Creating a dataframe from a csv
  df = pd.read_csv(file_path)
  
  
  """
  Using Plotly Express
  """
  #Creating the figure, pandas dataframe is one of the arguments
  #x is the "Petal Length" column
  #y is the "Petal Width" column
  #color is picked based on the "Class" column
  fig_1 = px.scatter(df, x="Petal Length", y="Petal Width", color="Class")
  
  #Updating the layout to change the title of the figure
  fig_1.update_layout(title="Iris Dataset")
  
  #displaying the figure
  fig_1.show()
  
  """
  Using Plotly Graph Objects
  """
  #Creating a plotly graph object
  fig_2 = go.figure()
  
  #Updating the titles of the figure
  fig_2.update_layout(xaxis_title="number of observations",yaxis_title="posterior probability",title="Single bag of h3 candy")
  
  #Adding a trace to the figure
  fig_2.add_trace(go.Scatter(x = observations, y = h1_p1, mode='lines', name='p(h1|d)'))
  
  #Displaying the graph
  fig_2.show()
  
  
  
if __name__ == "__main__":
  main()
  
  
