from datetime import datetime

import scrapy

from itemloaders.processors import TakeFirst, MapCompose


def convert_date(date_string):
    date = datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S%z')
    return date

def html_to_text(html):
    selector = scrapy.Selector(text=html)
    text_list = selector.xpath('//text()').getall()
    text = ' '.join(text_list)
    text = text.strip()
    return text

class MtavaritvItem(scrapy.Item):
    date = scrapy.Field(
        input_processor=MapCompose(convert_date),
        output_processor=TakeFirst()
    )
    title = scrapy.Field(
        input_processor=MapCompose(str.strip),
        output_processor=TakeFirst()
    )
    content_url = scrapy.Field(
        output_processor=TakeFirst()
    )
    content = scrapy.Field(
        input_processor=MapCompose(html_to_text),
        output_processor=TakeFirst()
    )
