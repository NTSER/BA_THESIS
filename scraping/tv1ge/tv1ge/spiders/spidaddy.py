import json

import scrapy
from scrapy.loader import ItemLoader

from ..items import Tv1GeItem
from ..utils import URLS, get_next_url


class SpidaddySpider(scrapy.Spider):
    name = "spidaddy"
    allowed_domains = ["1tv.ge"]
    start_urls = URLS

    def start_requests(self):
        for url in URLS:
            yield scrapy.Request(url, method='POST', callback=self.parse)

    def parse(self, response):
        json_resp = json.loads(response.body)
        items = json_resp.get('data')
        page_exists = (items != "no ids")
        if page_exists:
            links = [item.get('post_permalink') for item in items]
            yield from response.follow_all(links, callback=self.parse_content)

            next_url = get_next_url(response.url)
            yield response.follow(next_url, callback=self.parse, method="POST")

    def parse_content(self, response):
        loader = ItemLoader(Tv1GeItem(), selector=response)
        loader.add_xpath('date', "//div[@class='article-date']/text()")
        loader.add_xpath('title', "//div[contains(@class, 'article-title')]/text()")
        loader.add_value('content_url', response.url)
        loader.add_xpath('content', "//div[contains(@class, 'article-intro')]//p//text()")

        yield loader.load_item()
