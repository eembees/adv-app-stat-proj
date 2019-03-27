import scrapy
from scrapy_splash import SplashRequest

import json

# set some lists here for future reference
link = 'http://karakterstatistik.stads.ku.dk/#searchText=&term=Summer-2015&block=&institute=&faculty=2694&searchingCourses=true&page=1'

facDict = {
    'HUM':2920,
    'JUR':1988,
    'SCIENCE':1868,
    'SAMF': 2710,
    'SUND':2743,
    'TEO':2694
}

term_list = [
    'Sommer 2014',
    'Vinter 2014',
    'Sommer 2015',
    'Vinter 2015',
    'Sommer 2016',
    'Vinter 2016',
    'Sommer 2017',
    'Vinter 2017',
    'Sommer 2018',
    'Vinter 2018',
]

api_url = 'karakterstatistik.stads.ku.dk/#searchText=&term=Summer-2015&block=&institute=&faculty=1868&searchingCourses=true&page={}'


class kuItem(scrapy.Item):
    # define the fields for your item here like:
    link = scrapy.Field()
    attr = scrapy.Field()

class kuSpider(scrapy.Spider):
    name = 'ku_spider'
    # start_urls = ['http://karakterstatistik.stads.ku.dk/#searchText=&term=Summer-2015&block=&institute=&faculty=1868&searchingCourses=true&page=1']
    # api_url = 'http://karakterstatistik.stads.ku.dk/#searchText=&term=Summer-2015&block=&institute=&faculty=1868&searchingCourses=true&page={}'
    start_urls = [api_url.format(1)]

    custom_settings = {
        'SPLASH_URL': 'http://localhost:8050',
        'DOWNLOAD_DELAY' : '0.25',

    # if installed Docker Toolbox:
        #  'SPLASH_URL': 'http://192.168.99.100:8050',
        'DOWNLOADER_MIDDLEWARES': {
            'scrapy_splash.SplashCookiesMiddleware': 723,
            'scrapy_splash.SplashMiddleware': 725,
            'scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware': 810,
        },
        'SPIDER_MIDDLEWARES': {
            'scrapy_splash.SplashDeduplicateArgsMiddleware': 100,
        },
        'DUPEFILTER_CLASS': 'scrapy_splash.SplashAwareDupeFilter',
    }

    def start_requests(self):
        yield SplashRequest(
            url=api_url.format(1),
            callback=self.parse,
        )

    def parse(self, response):
        resultTable = response.xpath('//table[@class="searchResultTable"]')

        for linkitem in resultTable.xpath('//a'):
            yield linkitem.xpath('@href')

        # data = json.loads(response.text)
        # yield data

