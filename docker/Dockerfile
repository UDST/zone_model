FROM us.gcr.io/swarm-test-1470707908646/udst_base

RUN git clone https://456d1a9002992943f5045b7e1a2d25eeb2589371:x-oauth-basic@github.com/janowicz/orca
RUN cd orca && python setup.py develop

RUN pip install -e git+https://456d1a9002992943f5045b7e1a2d25eeb2589371:x-oauth-basic@github.com/udst/urbansim.git#egg=urbansim

#RUN git clone https://456d1a9002992943f5045b7e1a2d25eeb2589371:x-oauth-basic@github.com/urbansim/modeldev_pipeline.git
#RUN mkdir ./modeldev_pipeline/temp
#RUN git clone https://456d1a9002992943f5045b7e1a2d25eeb2589371:x-oauth-basic@github.com/urbansim/block_model.git ./modeldev_pipeline/temp

RUN pip install -e git+https://456d1a9002992943f5045b7e1a2d25eeb2589371:x-oauth-basic@github.com/urbansim/autospec.git#egg=autospec
RUN pip install -e git+https://456d1a9002992943f5045b7e1a2d25eeb2589371:x-oauth-basic@github.com/urbansim/autospec_recipes.git#egg=autospec_recipes
RUN pip install -e git+https://456d1a9002992943f5045b7e1a2d25eeb2589371:x-oauth-basic@github.com/urbansim/variable_generators.git#egg=variable_generators
RUN pip install -e git+https://456d1a9002992943f5045b7e1a2d25eeb2589371:x-oauth-basic@github.com/urbansim/autocalib.git#egg=autocalib
RUN pip install -e git+https://456d1a9002992943f5045b7e1a2d25eeb2589371:x-oauth-basic@github.com/urbansim/demographic_data.git#egg=adskdemographicdataauto
RUN pip install -e git+https://456d1a9002992943f5045b7e1a2d25eeb2589371:x-oauth-basic@github.com/urbansim/orca_test.git#egg=orca_test

WORKDIR /

RUN git clone https://456d1a9002992943f5045b7e1a2d25eeb2589371:x-oauth-basic@github.com/urbansim/zone_model.git

WORKDIR /zone_model/zone_model/data

RUN wget https://storage.googleapis.com/urbansim/zone_model/model_data.h5



WORKDIR /zone_model/zone_model
