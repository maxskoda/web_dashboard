import threading
import time
from requests_html import HTMLSession
import re

url='http://dataweb.isis.rl.ac.uk/IbexDataweb/default.html?Instrument=inter'

session = HTMLSession()


def get_stuff():
    r = session.get(url)
    r.html.render()

    # #take the rendered html and find the element that we are interested in
    inst_pvs = r.html.find('#inst_pvs', first=True).text
    main = r.html.find('#groups', first=True).text
    inst_pvs = r.html.find('#inst_pvs', first=True).text
    m = re.search('(?<=Inst. Time:).*(?=:)', inst_pvs)
    # lang_bar = r[0].html.find('#LangBar', first=True)
    print("First", m.group(0))

get_stuff()

var = False


def f():
    import time
    counter = 0
    while var:
        time.sleep(1)
        get_stuff()
        print("Function {} run!".format(counter))
        counter+=1


t1 = threading.Thread(target=f)

print("Starting thread")
var = True
t1.start()
time.sleep(10)
print("Something done")
var = False
t1.join()
print("Thread Done")