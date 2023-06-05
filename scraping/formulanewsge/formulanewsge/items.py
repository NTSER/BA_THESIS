from datetime import datetime

import scrapy

from itemloaders.processors import TakeFirst, Join, MapCompose



def html_do_date(html):
    month_to_num = {
        'იან':'01', 'თებ':'02', 'მარ':'03', 'აპრ':'04', 'მაი':'05', 'ივნ':'06',
        'ივლ':'07', 'აგვ':'08', 'სექ':'09', 'ოქტ':'10', 'ნოე':'11', 'დეკ':'12'
    }
    selector = scrapy.Selector(text=html)
    date_list = selector.xpath('.//text()').getall()
    date_list = [date.replace(',', '').strip() for date in date_list]
    date_string = ' '.join(date_list)
    date_list = date_string.split()
    date_list[1] = month_to_num.get(date_list[1])
    date_string = ' '.join(date_list)
    date = datetime.strptime(date_string, "%d %m %Y %H:%M")

    return date

def join_and_strip(values):
    joined_values = Join()(values)
    stripped_values = joined_values.strip()
    return stripped_values

def content_from_quote(quote, quote_description):
    pass

class FormulanewsgeItem(scrapy.Item):
    date = scrapy.Field(
        input_processor=MapCompose(str.strip, html_do_date),
        output_processor=TakeFirst()
    )
    title = scrapy.Field(
        input_processor=MapCompose(str.strip),
        output_processor=Join()
    )
    content_url = scrapy.Field(
        output_processor=TakeFirst()
    )
    content = scrapy.Field(
        input_processor=MapCompose(str.strip),
        output_processor=join_and_strip
    )
    
