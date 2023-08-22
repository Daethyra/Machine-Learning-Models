# World Data Preprocessing Module Updates

`(Creation Date: 8 / 22 / 2023)`

Today, we made important updates to the `preprocessing.py` file. We added new features to handle missing data and convert specific columns into a format we can work with.

## What We Did

### 1. Filling Missing Data

We added a method called `impute_missing_values_with_knn` to the `DataPreprocessor` class. This method finds and fills in missing numbers in the data using a technique called k-nearest neighbors. It helps keep the data's patterns intact.

### 2. Changing Currency and Percentages

We also added a method called `convert_currency_and_percentage_columns`. It changes columns that have money symbols like `$` or percentage signs like `%` into regular numbers.

## How to Use It

First, you need to create a `DataPreprocessor` object. Then, you can use the new methods to change the data. Here's an example:

<pre><div class="bg-black rounded-md mb-4"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>python</span><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-python">preprocessor = DataPreprocessor()
data = preprocessor.impute_missing_values_with_knn(data)
data = preprocessor.convert_currency_and_percentage_columns(data)
</code></div></div></pre>

## Extra Info

* **Missing Values** : Check out how the method `impute_missing_values_with_knn` works in the code to see how it fills in missing values.
* **Currency and Percentages** : Look at `convert_currency_and_percentage_columns` to see how it changes money and percentage columns.

## Conclusion

These updates make the preprocessing part of our code better. It's now easier to work with the data and get it ready for analysis. If you want to know more about how everything works, you can look at the code comments
