def factorial(n):
    result=1
    for i in range(1,n+1):
        result*=i
    return result

def lambda_handler(event, context):
    factorial(10000)
    event["map_list"] = list(range(10))
    return {
        'statusCode': 200,
        'body': {"name":"f7","input":event}
    }