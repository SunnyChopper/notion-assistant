from typing import Dict, List
import requests
import json
import os

from dotenv import load_dotenv

load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
ROOT_PAGE_ID = os.getenv("ROOT_PAGE_ID")

class NotionChildPage:
	'''
	Represents a child page from Notion.
	'''
	title: str = ''
	page_id: str = ''

	def __init__(self, title: str, page_id: str):
		self.title = title
		self.page_id = page_id

	def __str__(self) -> str:
		return json.dumps(self.__dict__)

class NotionPage:
	'''
	Represents a page from Notion.
	'''
	page_id: str = ''
	content: Dict = {}
	full_content: str = ''
	child_pages: List[NotionChildPage] = []

	def __init__(self, page_id: str, content: Dict, full_content: str, child_pages: List[NotionChildPage]):
		self.page_id = page_id
		self.content = content
		self.full_content = full_content
		self.child_pages = child_pages

class NotionReader:
	'''
	Read the content of a page from Notion.
	'''
	headers = {
		"Authorization": f"Bearer {NOTION_TOKEN}",
		"Notion-Version": "2022-06-28",
		"Content-Type": "application/json"
	}

	@staticmethod
	def get_page_content(page_id) -> NotionPage:
		'''
		Get the content of a page from Notion.
		'''
		url = f"https://api.notion.com/v1/pages/{page_id}"
		response = requests.get(url, headers=NotionReader.headers)
		if response.status_code == 200:
			page_data = response.json()
			content = {}
			for key, value in page_data['properties'].items():
				if 'title' in value:
					content[key] = [text['plain_text'] for text in value['title']]
				elif 'rich_text' in value:
					content[key] = [text['plain_text'] for text in value['rich_text']]
				elif 'url' in value:
					content[key] = value['url']
				elif 'select' in value:
					content[key] = value['select']['name'] if value['select'] else None
				elif 'multi_select' in value:
					content[key] = [option['name'] for option in value['multi_select']]
				elif 'date' in value:
					content[key] = value['date']['start'] if value['date'] else None
				elif 'people' in value:
					content[key] = [person['name'] for person in value['people']]
				elif 'relation' in value:
					content[key] = [relation['id'] for relation in value['relation']]
				elif 'checkbox' in value:
					content[key] = value['checkbox']
				elif 'number' in value:
					content[key] = value['number']
				# Add more property types here as needed

			# Fetch block content for richer text
			blocks: list = NotionReader.get_page_blocks(page_id)
			full_content: str = NotionReader.process_blocks(blocks)
			child_pages: list = NotionReader.get_child_pages(blocks)
			
			notion_page = NotionPage(page_id=page_id, content=content, full_content=full_content, child_pages=child_pages)
			return notion_page
		else:
			# Print full error details
			print(f"Error Status Code: {response.status_code}")
			try:
				error_json = response.json()
				print(f"Full Error Response: {json.dumps(error_json, indent=2)}")
			except ValueError:
				# If the response content is not JSON, print as text
				print(f"Error Response Content: {response.text}")
			raise Exception(f"Failed to retrieve page content. Status code: {response.status_code}")

	@staticmethod
	def get_page_blocks(page_id) -> list:
		'''
		Get the blocks of a page from Notion.
		'''
		url = f"https://api.notion.com/v1/blocks/{page_id}/children"
		response = requests.get(url, headers=NotionReader.headers)
		if response.status_code == 200:
			return response.json()['results']
		else:
			print(f"Error fetching blocks. Status Code: {response.status_code}")
			try:
				error_json = response.json()
				print(f"Full Error Response for blocks: {json.dumps(error_json, indent=2)}")
			except ValueError:
				print(f"Error Response Content for blocks: {response.text}")
			return []

	@staticmethod
	def process_blocks(blocks) -> str:
		'''
		Process the blocks of a page from Notion.
		'''
		full_content = []
		for block in blocks:
			block_type = block['type']
			if block_type == 'paragraph':
				text = ' '.join([text['plain_text'] for text in block[block_type]['rich_text']])
				full_content.append(text)
			elif block_type in ['heading_1', 'heading_2', 'heading_3']:
				text = block[block_type]['rich_text'][0]['plain_text'] if block[block_type]['rich_text'] else ""
				full_content.append(f"{'#' * int(block_type[-1])} {text}")
			elif block_type == 'bulleted_list_item':
				text = ' '.join([text['plain_text'] for text in block[block_type]['rich_text']])
				full_content.append(f"- {text}")
			elif block_type == 'numbered_list_item':
				text = ' '.join([text['plain_text'] for text in block[block_type]['rich_text']])
				full_content.append(f"1. {text}")
			elif block_type == 'child_page':
				title: str = block['child_page']['title']
				page_id: str = block['parent']['page_id']
				text = f'- [{title}]({page_id})'
				full_content.append(text)
			else:
				full_content.append(str(block))
		
		return '\n'.join(full_content)

	@staticmethod
	def get_child_pages(blocks) -> List[NotionChildPage]:
		'''
		Get the child pages of a page from Notion.
		'''
		child_pages: List[NotionChildPage] = []
		for block in blocks:
			if block['type'] == 'child_page':
				title: str = block['child_page']['title']
				page_id: str = block['id']
				child_page = NotionChildPage(title=title, page_id=page_id)
				child_pages.append(child_page)
		return child_pages


if __name__ == "__main__":
	try:
		notion_page = NotionReader.get_page_content(ROOT_PAGE_ID)
		print("Page Properties:", notion_page.content)
		print("Page Blocks Content:", notion_page.full_content)
		print("Child Pages:", notion_page.child_pages)
	except Exception as e:
		print(e)