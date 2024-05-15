import urllib, urllib.request

def bib_to_id(title_str = '', authors_str = '', start_idx=0, max_idx=1):
    """
    This func takes a title as a string, a list of the authors, which can be empty, and the start and max indexes for the search. 
    It returns the arXiv id of the paper if it exists and there exists only one match for the API search, and None otherwise.
    """

    def get_ids_from_url(url):
        try:
            data = urllib.request.urlopen(url)
        except Exception as e:
            print(str(e) + ' in get_ids_from_url() in Bib_to_id.py')
            return None
        data_str = data.read().decode('utf-8')
        data_items=data_str.split('<entry>')[1:]

        data_items_ids = [item.split('<id>http://arxiv.org/abs/')[1].split('<')[0] for item in data_items]
        return data_items_ids

    url = 'http://export.arxiv.org/api/query?search_query=ti:%22'  + title_str.replace(" ", "+").replace("\n","+") + '%22&start=' + str(start_idx) + '&max_results=' + str(max_idx)
    
    ids = get_ids_from_url(url)
    #if ids == None or len(ids) != 1:
    #    url = 'http://export.arxiv.org/api/query?search_query=au:' + authors_str.replace(" ", "+").replace("\n","+") + '+AND+ti:%22' + title_str.replace(" ", "+").replace("\n","+") + '%22&start=' + str(start_idx) + '&max_results=' + str(max_idx)
    #    ids = get_ids_from_url(url)
    
    return None if ids == None or len(ids) != 1 else ids[0]



# Towards a rigorous science of interpretable machine learning