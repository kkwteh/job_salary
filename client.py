import json
import requests

def send_prediction_request():
    request = {'Category': 'IT Jobs',
               'Company': 'Yolk Recruitment',
               'ContractTime': 'permanent',
               'ContractType': 'full_time',
               'FullDescription': 'A leading Medical Device company based in Oxfordshire is looking for the next Toolmaker to join their team to assist with the manufacturing team across the site in line with a continued period of growth. The responsibilities of the Toolmaker are as follows:  Management and updates of database  Managing the PPM schedules in line with customer expectations  Development, manufacture and maintenance of appropriate tooling  Effective fault diagnostics and management whilst encouraging a continuous improvement ethos across the site To fulfil these requirements for the toolmaker you must have the following skills/experience:  Educated to HNC level/equivalent experience  Strong manufacturing background  Experience with plastic injection tool design and manufacture  Use of Fault finding and diagnostics would be an advantage  Excellent communication and presentation skills The successful Toolmaker will receive a competitive salary and have the chance to work in a very reputable and rapidly growing company within the medical device industry. To Apply Please forward your most uptodate CV together with salary details quoting ref ****/TG or by calling Tom Gorton on **** **** **** JAM Recruitment Ltd is acting as an Employment Agency in relation to this vacancy. View our latest jobs today at www.jamrecruitment.co.uk and follow us on Facebook, Twitter & LinkedIn JAM Recruitment is acting as an employment agency with regards to this position',
               'LocationNormalized': 'Oxford',
               'LocationRaw': 'Oxford Oxfordshire South East',
               'SourceName': 'technojobs.co.uk',
               'Title': 'Manufacturing Toolmaker',
               }
    print json.dumps(request)
    resp = requests.post('http://localhost:5000/predict',
                         json.dumps(request))
    print json.loads(resp.content)

if __name__ == '__main__':
    send_prediction_request()
