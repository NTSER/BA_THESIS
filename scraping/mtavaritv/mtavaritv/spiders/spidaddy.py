import json

import scrapy

from ..items import MtavaritvItem
from mtavaritv.utils import URLS

from scrapy.loader import ItemLoader


class SpidaddySpider(scrapy.Spider):
    name = "spidaddy"
    allowed_domains = ["mtavari.tv"]
    start_urls = URLS

    def parse(self, response):
        json_resp = json.loads(response.body)
        items = json_resp['data']

        for item in items:
            loader = ItemLoader(item=MtavaritvItem(), response=response)
            content_url = item.get('links').get('self').get('href')

            loader.add_value('date', item.get('attributes').get('created'))
            loader.add_value('title', item.get('attributes').get('title'))
            loader.add_value('content_url', content_url)

            yield response.follow(url=content_url, callback=self.parse_content, meta={'loader':loader})

        next_page = json_resp.get('links').get('next')
        if next_page is not None:
            yield response.follow(next_page['href'], callback=self.parse)

    def parse_content(self, response):
        loader = response.meta['loader']
        json_resp = json.loads(response.body)
        html = json_resp.get('data').get('attributes').get('body')
        loader.add_value('content', html)

        yield loader.load_item()