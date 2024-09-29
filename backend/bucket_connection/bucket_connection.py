from flask import Flask, request
import boto3

AWS_ACCESS_KEY_ID = 'your_aws_access_key_id'
AWS_SECRET_ACCESS_KEY = 'your_aws_secret_access_key'
BUCKET_NAME = 'your_s3_bucket_name'


def connect_bucket():
  s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
  return s3, BUCKET_NAME