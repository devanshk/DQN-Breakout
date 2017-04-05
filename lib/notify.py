import smtplib
import os
import json
import numpy as np
from lib.hyperparams import *
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import datetime as dt


def minutes(td):
    return (td.seconds//60)%60

def send_epoch_email(epoch, rewards, epoch_durations):
    with open('notify.json') as data_file:
        receivers = json.load(data_file)
    curr_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    avg_reward = round(np.mean(rewards), 2)
    avg_epoch_duration = np.mean(epoch_durations)
    seconds_left = int(avg_epoch_duration) * (EPOCHS - epoch)
    mm, ss = divmod(seconds_left, 60)
    hh, mm = divmod(mm, 60)
    subject = "Training Update".format(epoch)
    body = "Just updating you on the training process. \
    The time is {}. I will update you every {} epochs. This is epoch {}, {} more epochs to go!\
    Given my current pace, I will probably be done training in {} hours and {} minutes.\
    Here are some stats to let you know how well I am doing:\n\n".format(\
    curr_time, NOTIFY_RATE, epoch, EPOCHS - epoch, hh, mm)
    data = {"Avg. reward":avg_reward,
            "Avg. epoch duration (minutes)":round(avg_epoch_duration/60, 2)}

    send(subject=subject,
            data = data,
            body = body,
            to = receivers["emails"],
            attach = "data/tmp/avg_reward.png")



def send(subject="Training Update",
                to = [os.environ["GM_ADDRESS"]],
                body = "Hello, here is an update",
                data = {"time":dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
                attach = None
                ):

    fromaddr = os.environ["GM_ADDRESS"]
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, os.environ["GM_PASS"])

    for person in to:
        fromaddr = os.environ["GM_ADDRESS"]
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(fromaddr, os.environ["GM_PASS"])
        msg = MIMEMultipart()

        msg['From'] = fromaddr
        msg['To'] = person
        msg['Subject'] = subject

        for k,v in data.items():
            body += "{} : {}\n".format(k,v)

        msg.attach(MIMEText(body, 'plain'))

        if attach is not None:
            filename = attach
            attachment = open(os.environ["PWD"] + "/" + filename, "rb")

            part = MIMEBase('application', 'octet-stream')
            part.set_payload((attachment).read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

            msg.attach(part)

        text = msg.as_string()
        server.sendmail(fromaddr, person, text)
        server.quit()
