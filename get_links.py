import itertools

import time
import pandas as pd

from selenium import webdriver

from selenium.common.exceptions import ElementNotVisibleException
from selenium.common.exceptions import NoSuchElementException


# stuff goes here

api_link = 'http://karakterstatistik.stads.ku.dk/#searchText=&term={}&block=&institute=&faculty={}&searchingCourses=true&page=1'
api_output = 'data/links_{}_{}.txt'
facDict = {
    'HUM': 2920,
    'JUR': 1988,
    'SCIENCE': 1868,
    'SAMF': 2710,
    'SUND': 2743,
    'TEO': 2694
}

term_list = [
    'Summer-2014',
    'Winter-2014',
    'Summer-2015',
    'Winter-2015',
    'Summer-2016',
    'Winter-2016',
    'Summer-2017',
    'Winter-2017',
    'Summer-2018',
    'Winter-2018',
]

facList = [key for key in facDict]

# tupcombos = zip(facList, term_list)

fac_term_combinations = itertools.product(term_list, facList)

for term, fac in fac_term_combinations:
    # make the right link to start from, and output file
    start_url = api_link.format(term, facDict[fac])
    outputName = api_output.format(fac, term)

    print('Now starting on: ')
    print(fac)
    print(term)

    # open new driver each time to be sure
    driver = webdriver.Chrome()
    driver.get(start_url)

    # get results
    searchbutton = driver.find_elements_by_xpath('//*[@value="Søg"]')[0]
    searchbutton.click()
    time.sleep(1)

    # get a list to store all links found
    link_list = []

    # at one point we get an ElementNotVisibleException or NoSuchElementException from this item
    try:
        next_button = driver.find_element_by_link_text('Næste side')

        while len(next_button.get_attribute('style')) == 0:
            # find all links of search, it doesn't matter if not all of them are perfect, we can clean them up later
            search_links = driver.find_elements_by_xpath('//a')
            for a in search_links:
                try:
                    link_list.append(a.get_attribute('href'))
                except TypeError:
                    pass

            print("Proceeding to next page")
            next_button.click()
            time.sleep(1)
        else:
            print("Now we're on the last page")
            # find all links of search, it doesn't matter if not all of them are perfect, we can clean them up later
            search_links = driver.find_elements_by_xpath('//a')
            for a in search_links:
                try:
                    link_list.append(a.get_attribute('href'))
                except TypeError:
                    pass


    except NoSuchElementException:
        print('This one has only one page')
        search_links = driver.find_elements_by_xpath('//a')
        for a in search_links:
            try:
                link_list.append(a.get_attribute('href'))
            except TypeError:
                pass

    with open(outputName, 'w') as f:
        for line in [l for l in link_list if l is not None]:
            f.write(line)
            f.write('\n')
    # close the webview
    driver.close()
exit()
