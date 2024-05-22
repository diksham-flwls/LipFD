import boto3

bucket_name = "flwls-staging-eu-west-1-research-centipede"
folder_name = "centipede/diksha/shared_datasets/talking_heads/voxceleb/repo/"
client = boto3.client('s3')

def list_subdir(bucket_name, subdir_path, delimiter="/"):
    # its important to have the prefix ending with / to list files
    subdir_path = subdir_path + "/" if subdir_path[-1] != "/" else subdir_path

    paginator = client.get_paginator('list_objects')
    
    if delimiter:
        pagination = paginator.paginate(Bucket=bucket_name, Prefix=subdir_path, Delimiter=delimiter) 
    else:
        pagination = paginator.paginate(Bucket=bucket_name, Prefix=subdir_path)

    for result in pagination:
        for prefix in result.get('CommonPrefixes'):
            yield prefix.get('Prefix')

def list_files(bucket_name, subdir_path):
    subdir_path = subdir_path + "/" if subdir_path[-1] != "/" else subdir_path
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=subdir_path):
        for content in result['Contents']:
            yield content['Key']

def download_file(bucket_name, file_path, save_path):
    BUCKET_RESOURCE = boto3.resource("s3").Bucket(bucket_name)
    BUCKET_RESOURCE.download_file(
        file_path,
        str(save_path),
    )

def upload_file(bucket_name, local_file_path, target_location):
    BUCKET_RESOURCE = boto3.resource("s3").Bucket(bucket_name)
    BUCKET_RESOURCE.upload_file(local_file_path, target_location)

# l = [item for item in list_files("flwls-sdp-eu-west-1-external-restricted", "voxceleb/v2/test/mp4/mp4/id00017/01dfn2spqyE/")]
# print(l)