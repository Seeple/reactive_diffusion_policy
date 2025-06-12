from multiprocessing import Process
import zerorpc

def worker(id):
    client = zerorpc.Client()  
    client.connect("tcp://192.168.2.187:5000")
    # print(client.echo(f"Hello from {id}"))
    while True:
        print(id, client.get_robot_state())

if __name__ == "__main__":
    processes = [Process(target=worker, args=[id]) for id in range(4)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()