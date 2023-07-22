import paho.mqtt.client as paho
import ssl

# ---------------------------------------MQTT Functions-----------------------------------------
flag_connected = 0
def on_connect(client, userdata, flags, rc):  
    global flag_connected
    flag_connected = 1 
    print("Connected to the cloud!")

def on_disconnect(client, userdata, rc):
    global flag_connected
    flag_connected = 0 
    print("DISCONNECTED!")

    # Defining the mqtt connection  
client = paho.Client() 
client.on_connect = on_connect
client.on_disconnect = on_disconnect

    # Setting the username password
client.username_pw_set(username='ngalande', password='alprs@pappi')

    # Connecting to the broker  
client.tls_set(cert_reqs=ssl.CERT_NONE, tls_version=ssl.PROTOCOL_TLS)
client.connect("4eb4a74af4a64aa7b440dd2d2451e924.s2.eu.hivemq.cloud", 8883)
#client.loop_forever()
#---------------------------------------------------------------------------------------------

client.subscribe("alprs/plate", 1)

# This is the Subscriber

def on_connect(client, userdata, flags, rc):
  print("Connected with result code "+str(rc))
  client.subscribe("alprs/plate", 1)


#------------------------------------------------------
import pika
import os

# Access the CLOUDAMQP_URL environment variable and parse it (fallback to localhost)
url = os.environ.get('test', 'amqps://sifwmpwt:Nx2yPN91PzEcUVtBETaRxShmMq7_rM57@sparrow.rmq.cloudamqp.com/sifwmpwt')
params = pika.URLParameters(url)
connection = pika.BlockingConnection(params)
channel = connection.channel() # start a channel
channel.queue_declare(queue='etoll') # Declare a queue
# channel.basic_publish(exchange='',
#                       routing_key='etolldata',
#                       body='Test from Blegga')

# print(" [x] Sent Information about image scaling")
# connection.close()
#-----------------------------------------------------
def on_message(client, userdata, msg):
    plate = msg.payload.decode()
    print(plate)
    #client.publish("callback", payload=plate, qos=1)
    channel.basic_publish(exchange='',
                      routing_key='etoll',
                      body=plate)

   
    
#client.on_connect = on_connect
client.on_message = on_message

client.loop_forever()