# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor

from scrapy_splash import SplashRequest


class QuotesSpider(scrapy.Spider):
    name = "linkspider"
    allowed_domains = ['karakterstatistik.stads.ku.dk/']
    start_urls = ['karakterstatistik.stads.ku.dk/#searchText=&term=Summer-2015&block=&institute=&faculty=1868&searchingCourses=true&page=1']

    # http_user = 'splash-user'
    # http_pass = 'splash-password'

    def parse(self, response):
        le = LinkExtractor()
        for link in le.extract_links(response):
            yield SplashRequest(
                link.url,
                self.parse_link,
                endpoint='render.json',
                args={
                    'har': 1,
                    'html': 1,
                }
            )

    def parse_link(self, response):
        print("PARSED", response.real_url, response.url)
        print(response.css("title").extract())
        print(response.data["har"]["log"]["pages"])
        print(response.headers.get('Content-Type'))