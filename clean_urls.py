from pathlib import Path

dir_in = Path().cwd() / 'data'

'''
We need to remove all the entries that are not courses.
All courses have the term "Histogram" in the url
'''

files_urls = sorted([fn for fn in dir_in.glob('*.txt')])

for fn in files_urls:
    path_out = fn.parent.parent / 'urls' / fn.name.replace('links','urls')
    urls = fn.open().readlines()
    urls = [url.replace('\n','') for url in urls if 'Histogram' in url]

    path_out.touch(exist_ok=True)

    with path_out.open('w') as f:
        for url in urls:
            f.write(url+'\n')
