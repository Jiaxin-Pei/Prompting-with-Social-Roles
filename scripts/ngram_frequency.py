import requests 
import urllib 

def runQuery(query, start_year=2018, 
             end_year=2019, corpus=26,
             smoothing=0):
  
    # converting a regular string to 
    # the standard URL format
    # eg: "geeks for,geeks" will
    # convert to "geeks%20for%2Cgeeks"
    query = urllib.parse.quote(query)
  
    # creating the URL
    url = 'https://books.google.com/ngrams/json?content=' + query + '&year_start=' + str(start_year) + '&year_end=' + str(end_year) + '&corpus=' + str(corpus) + '&smoothing=' + str(smoothing) + ''
  
    # requesting data from the above url
    response = requests.get(url)
  
    # extracting the json data from the response we got
    output = response.json()
  
    # creating a list to store the ngram data
    return_data = []
  
    if len(output) == 0:
        # if no data returned from site,
        # print the following statement
        return "No data available for this Ngram."
    else:
        # if data returned from site,
        # store the data in return_data list
        for num in range(len(output)):
              # getting the name
            return_data.append((output[num]['ngram'], 
                                  
                                # getting ngram data
                                output[num]['timeseries']) 
                               ) 
    return return_data

role_path = "path/to/role/file"
role_df = pd.read_csv(role_path)
role_str = ', '.join(role_df['role'].astype(str))

ngram_full = runQuery(role_str)
ngram_freq_full = {key: np.mean(values) for key, values in ngram_full}
ngram_freq_df = pd.DataFrame(ngram_freq_full.items(), columns=['role', 'N-gram Frequency (2018-2019)'])

role_df = pd.merge(role_df, ngram_freq_df, how='left')

result_path = "path/to/result"
role_df.to_csv(result_path, index=False)