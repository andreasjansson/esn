FROM    ubuntu

RUN     echo "deb http://archive.ubuntu.com/ubuntu precise main universe" > /etc/apt/sources.list
RUN     apt-get update
RUN     apt-get -y install python2.7 python-setuptools python-numpy python-scipy python-matplotlib unzip
RUN     easy_install pip
RUN     pip install simplejson

RUN	mkdir /opt/esn
ADD	vocals /opt/esn/vocals
ADD	segments /opt/esn/segments
ADD     esn.py /opt/esn/esn.py
ADD	cma.py /opt/esn/cma.py
ADD	test_data.py /opt/esn/test_data.py
ADD	optimise_instrumentalness.py /opt/esn/optimise_instrumentalness.py

WORKDIR /opt/esn
CMD	python /opt/esn/optimise_instrumentalness.py
