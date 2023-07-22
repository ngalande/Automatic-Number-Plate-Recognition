import pika
import os

# Access the CLOUDAMQP_URL environment variable and parse it (fallback to localhost)
url = os.environ.get('test', 'amqps://sifwmpwt:Nx2yPN91PzEcUVtBETaRxShmMq7_rM57@sparrow.rmq.cloudamqp.com/sifwmpwt')
params = pika.URLParameters(url)
connection = pika.BlockingConnection(params)
channel = connection.channel() # start a channel
channel.queue_declare(queue='etolldata') # Declare a queue
channel.basic_publish(exchange='',
                      routing_key='etolldata',
                      body='Test from Blegga')

print(" [x] Sent Information about image scaling")
connection.close()
#amqps://sifwmpwt:Nx2yPN91PzEcUVtBETaRxShmMq7_rM57@sparrow.rmq.cloudamqp.com/sifwmpwt