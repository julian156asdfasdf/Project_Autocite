import urllib, urllib.request

def bib_to_id(title_str, start_idx_str, max_idx_str, authors_str = ''):
    """
    This func takes a title as a string, a list of the authors, which can be empty, and the start and max indexes for the search. 
    It returns the arXiv id of the paper if it exists and there exists only one match for the API search, and None otherwise.
    """
    url = 'http://export.arxiv.org/api/query?search_query=au:' + authors_str + '+AND+ti:%22' + title_str + '%22&start=' + start_idx_str + '&max_results=' + max_idx_str
    data = urllib.request.urlopen(url)
    data_str = data.read().decode('utf-8')
    data_items=data_str.split('<entry>')[1:]

    data_items_ids = [item.split('<id>http://arxiv.org/abs/')[1].split('<')[0] for item in data_items]
    
    return None if len(data_items_ids) != 1 else data_items_ids