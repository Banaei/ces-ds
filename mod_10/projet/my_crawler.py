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
hbase_table_name = 'wiki_test'

# Deleting the table
if hbase_table_name in connection.tables():
    connection.delete_table(hbase_table_name, disable=True)

# Create the test table, with a column family (cf)
connection.create_table(hbase_table_name, {'cf':{}})

# Openning the table
table = connection.table(hbase_table_name)

class WikiSpider(scrapy.Spider):
    
    
    name = 'WikipediaSpider' #identifiant du robot présenté au site crawlé
    start_urls = ['http://localhost']
    allowed_domains = ['localhost']
    custom_settings = {"DOWNLOAD_HANDLERS": {"s3": None}} # Pour éviter l'erreur bizarre de démarrage 
    url_count = 0

    def parse(self, response):
#        for content in response.xpath("//div[@id='bodyContent']//*[self::p or self::ul]//text()").extract():
        content = ' '.join(response.xpath("//div[@id='bodyContent']//*[self::p or self::ul]//text()").extract())
        
        data = content.encode('utf-8')
        table.put(response.url, {'cf:content': data})
        
        print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  ', self.url_count
            
        for href in response.xpath('//a/@href').extract():
            href = response.urljoin(href)
            if (href.startswith('http://localhost/articles') and ('%7E' not in href)):
                self.url_count += 1
                if (self.url_count>100):
                    break
                yield scrapy.Request(href, callback=self.parse)
                
                
    def closed(self, reason):
        print '******************* Closing  *************************'
        print 'Reason : ', reason
        print 'URL COUNT = ', self.url_count



# *********************************************************
