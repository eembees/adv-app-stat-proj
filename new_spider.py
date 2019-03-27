# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor
import time
from scrapy_splash import SplashRequest

from selenium import webdriver

start_urls = [
    'http://karakterstatistik.stads.ku.dk/#searchText=&term=Summer-2015&block=&institute=&faculty=1868&searchingCourses=true&page=1']


class FirstSpider(scrapy.Spider):
    name = "firstspider"
    # allowed_domains = ['karakterstatistik.stads.ku.dk/']
    start_urls = ['http://karakterstatistik.stads.ku.dk/#searchText=&term=Summer-2015&block=&institute=&faculty=1868&searchingCourses=true&page=1']

    def __init__(self):
        self.driver = webdriver.Chrome('/etc/chromedriver')

    def start_requests(self):
        for url in start_urls:
            url = str(url)
            yield SplashRequest(url,
                                self.parse,
                                endpoint='render.json',
                                args={'wait': 2})

    def parse(self, response):
        self.driver.get(response.url)
        # find and clock the searchbutton
        searchbutton = self.driver.find_elements_by_xpath('//*[@value="Søg"]')[0]
        searchbutton.click()
        time.sleep(1)

        # now extract all the links
        # links_a = self.driver.find_elements_by_xpath('//a')
        links_a = self.driver.find_elements_by_class_name('searchResultTable')

        # filename = 'Analyst.html'
        # with open(filename, 'wb') as f:
        #     # f.write(astuff)
        #     f.write(response.body)

        filename = 'links.txt'
        with open(filename, 'w') as f:
            for ai in links_a:
                try:
                    f.write(ai.get_attribute('href'))
                except TypeError:
                    f.write(' ')

