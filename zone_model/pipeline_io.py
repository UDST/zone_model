import boto
import boto3

import orca
from urbansim.utils import yamlio

def to_s3(f, target_name_on_s3):
    """
    Move specified file to Amazon S3.
    Parameters
    ----------
    f : str
        Filename of file to load to S3
    target_name_on_s3 : str
        S3 name (key) to store file as.
        
    Returns
    -------
    None
    """
    conn = boto.connect_s3()
    bucket = conn.get_bucket('synthpop-data2')
    print 'Loading %s to S3 at %s.' % (f, target_name_on_s3)
    key = bucket.new_key(target_name_on_s3)
    key.set_contents_from_filename(f)
    
def file_exists_on_s3(f):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('synthpop-data2')
    print 'Checking for the existence of %s on S3.' % f
    existing_files = [fl.key for fl in bucket.objects.all()]
    exists = f in existing_files
    if exists:
        print '%s already exists.' % f
    else:
        print '%s does not exist.' % f
    return exists

# Called after both fitting and calibration to send the resulting YAML files to S3
@orca.step('yaml_configs_to_s3')
def yaml_configs_to_s3(region_name, yaml_config_path):
    yaml_configs = yamlio.yaml_to_dict(str_or_buffer=yaml_config_path)
    config_filename = yaml_config_path.split('/')[-1]
    to_s3(yaml_config_path, '%s/%s' % (region_name, config_filename))

    for model_type in yaml_configs.keys():
        configs = yaml_configs[model_type]
        for cfg in configs:
            to_s3('configs/%s' % cfg, '%s/%s' % (region_name, cfg))
