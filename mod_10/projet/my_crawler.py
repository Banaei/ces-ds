# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:11:47 2015

@author: user
"""
import happybase
import scrapy

# hbase shell # /!\
# scan "table"
# count "table"
# help"

connection = happybase.Connection('localhost')

# Deleting the table
connection.delete_table('wiki', disable=True)

# Create the test table, with a column family
connection.create_table('wiki', {'cf':{}})

# Openning the table
table = connection.table('wiki')

class WikiSpider(scrapy.Spider):
    
    
    name = 'wiki'
    start_urls = ['http://localhost']
    url_count = 0
    stop_words = []
    with open('stop_words.txt') as f:
        stop_words = f.read().splitlines()


#    def parse(self, response):
#        print '******************* Starting  *************************'
#        print 'RESPONSE : ',response
#        hrefs = response.xpath('//a/@href').extract()
#        valid_hrefs = list()
#        for href in hrefs:
#            if (href.startswith('articles')):
#                valid_hrefs.append(href)
#                print 'HREF = : ',href
#        contents = response.xpath("//div[@id='bodyContent']//*[self::p or self::ul]//text()").extract()
#        print 'CONTENTS LENGTH = ' , len(contents)
#        valid_contents = list()
#        for content in contents:
#            if (content not in self.stop_words) and  (len(content.strip())>0):
#                valid_contents.append(content)
#                print 'CONTENT = ' , content

    def parse(self, response):
#        for content in response.xpath("//div[@id='bodyContent']//*[self::p or self::ul]//text()").extract():
        content = ' '.join(response.xpath("//div[@id='bodyContent']//*[self::p or self::ul]//text()").extract())
        
        data = content.encode('utf-8')
        table.put(response.url, {'cf:content': data})
            
        for href in response.xpath('//a/@href').extract():
            href = response.urljoin(href)
            if (href.startswith('http://localhost/articles') and (href not in '%7E')):
                self.url_count += 1
                yield scrapy.Request(href, callback=self.parse)
                
                
    def closed(self, reason):
        print '******************* Closing  *************************'
        print 'Reason : ', reason
        print 'URL COUNT = ', self.url_count


            
#        for href in response.xpath('//div[@id=’bodyContent’]//*[self::p or self::ul]//text()'):
#            print '******************* Dans la boucle  *************************'
#            full_url = response.urljoin(href.extract())
#            print 'url = ', full_url
#            yield scrapy.Request(full_url, callback=self.parse_question)

#    def parse_question(self, response):
#        yield {
#            'title': response.css('h1 a::text').extract()[0],
#            'votes': response.css('.question .vote-count-post::text').extract()[0],
#            'body': response.css('.question .post-text').extract()[0],
#            'tags': response.css('.question .post-tag::text').extract(),
#            'link': response.url,
#        }









#
#
#
#
#table.put('row-key', {'family:qual1': 'value1',
#                      'family:qual2': 'value2'})
#
#row = table.row('row-key')
#print row['family:qual1']  # prints 'value1'
#
#for key, data in table.rows(['row-key-1', 'row-key-2']):
#    print key, data  # prints row key and data for each row
#
#for key, data in table.scan(row_prefix='row'):
#    print key, data  # prints 'value1' and 'value2'
#
#row = table.delete('row-key')