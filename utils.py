import matplotlib.pyplot as plt

def create_histogram(df, column_name):
    # Create the histogram
    plt.hist(df[column_name], bins=100)  # you can adjust 'bins' as needed
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.title('Histogram of Text Lengths')
    plt.show()

def trim(df,column, interval):
    df = df[df[column] > interval[0]]
    df = df[df[column] < interval[1]]
    return df
    

