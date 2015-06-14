#! /usr/local/bin/python3
# -*- utf-8 -*-


"""
Preprocess the mongodb documents.
"""


import dateutil.parser

from pymongo import MongoClient


client = MongoClient()
db = client.xuetangx


db.object.update_all({'start': 'null'}, {'$unset': {'start': 1}})
db.object.update_all({'children': ''}, {'$unset': {'children': 1}})
db.object.create_index('module_id')
db.object.create_index('children')
db.object.create_index('course_id')
db.enroll_all.create_index('enrollment_id')
db.log_all.create_index('enrollment_id')
db.log_all.create_index('time')


for obj in db.object.find({'start': {'$exists': 1}}):
    db.object.update_one(
        {'_id': obj['_id']},
        {'$set': {'start': dateutil.parser.parse(obj['start'])}})


for obj in db.object.find({'children': {'$exists': 1}}):
    db.object.update_one(
        {'_id': obj['_id']},
        {'$set': {'children': obj['children'].split()}})


for enroll in db.enroll_all.find():
    db.log_all.update_many(
        {'enrollment_id': enroll['enrollment_id']},
        {'$set': {
            'username': enroll['username'],
            'course_id': enroll['course_id']
        }}
    )


for log in db.log_all.find({'time': {'$exists': 1}}):
    db.object.update_one(
        {'_id': log['_id']},
        {'$set': {'time': dateutil.parser.parse(log['time'])}})
