import scrapy
from glob import glob
# set some lists here for future reference
link = 'http://karakterstatistik.stads.ku.dk/#searchText=&term=Summer-2015&block=&institute=&faculty=2694&searchingCourses=true&page=1'

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

# useful function for later use
def hasNumbers(inputString):
     return any(char.isdigit() for char in inputString)


class CourseItem(scrapy.Item):
    # define the fields for your item here like:
    link = scrapy.Field()
    term = scrapy.Field()
    title = scrapy.Field()
    faculty = scrapy.Field()
    institute = scrapy.Field()

    bestaet = scrapy.Field()
    ikke_bestaet = scrapy.Field()
    ej_modt = scrapy.Field()

    g_12 = scrapy.Field()
    g_10 = scrapy.Field()
    g_7 = scrapy.Field()
    g_4 = scrapy.Field()
    g_02 = scrapy.Field()
    g_00 = scrapy.Field()
    g_n3 = scrapy.Field()


class PageSpider(scrapy.Spider):
    name = 'pagespider'
    filename = 'urls/urls_HUM_Summer-2014.txt'
    # def __init__(self):

    def __init__(self, filename=None):
        read_urls = []
        filenames = glob('urls/*.txt')
        for filename in filenames:
            with open(filename, 'r') as f:
                read_urls.append([url.replace('\n','') for url in f.readlines()])

        self.start_urls = [item for sublist in read_urls for item in sublist]


    def parse(self, response):
        course = CourseItem()

        course['term'] = str(response).split('/')[5].replace('>','')
        course['title'] = response.xpath('//form//h2/text()').get().replace('\r\n', '').strip()
        course['faculty'] = response.xpath('//form//table')[0].xpath('//*[contains(text(),"Fakultet")]').getall()[1]\
            .replace('\r\n', '').replace('</td>','').replace('<td>','').strip()
        try:
            course['institute'] = response.xpath('//form//table')[0].xpath('//*[contains(text(),"Institut")]').getall()[1]\
                .replace('\r\n', '').replace('</td>','').replace('<td>','').strip()
        except IndexError:
            pass

        # tgrad = response.xpath('//form/table/tr[last()]/td[1]/table')
        tres = response.xpath('//form/table/tr[last()]/td[1]/table/tr/td/table')
        tres_data = tres.xpath('tr[position()>1]')


        for row in tres_data:
            row_data =   [s.replace('\r\n', '').strip().replace(' ','_').replace('-','n').replace('ø','o').replace('å','a').lower() for s in row.xpath('td[position()<3]/text()').getall()]
            if hasNumbers(row_data[0]):
                row_data[0] = 'g_{}'.format(row_data[0])
            course[row_data[0]] = row_data[1]


        yield course