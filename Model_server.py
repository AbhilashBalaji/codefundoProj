#!/usr/bin/env python
# coding: utf-8

# In[6]:


from azureml.core import Workspace

subscription_id ='ca64cab5-4953-4dec-b4c9-3a934974109d'
resource_group ='ML1'
workspace_name = 'MachineLearning'

try:
   ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
   ws.write_config()
   print('Library configuration succeeded')
except:
   print('Workspace not found')

from azureml.core import Workspace
ws = Workspace.from_config()

from azureml.core.model import Model

model_name = "Model_hurrthreat"
model = Model.register(model_path="model.pkl",
                        model_name=model_name,
                        tags={"data": "IRENE", "model": "classification"},
                        description="hurrthreat prediction",
                        workspace=ws)


# In[7]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
 
import azureml
from azureml.core import Workspace, Run

# display the core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)


# In[9]:


from azureml.core import Workspace
from azureml.core.model import Model

ws = Workspace.from_config()
model=Model(ws, 'Model_hurrthreat')
model.download(target_dir = '.')
import os 
# verify the downloaded model file
os.stat('model.pkl')


# In[18]:


import pickle
from sklearn.externals import joblib

#cols = ['Lat', 'Lon', 'pressure', 'wavedirection', 'waveheight']
clf = joblib.load('model.pkl')
arr=np.array([5.46,63.35,1015.2,989,2.9])
arr=arr.reshape(1,-1)
y_hat = clf.predict(arr)


# In[19]:


get_ipython().run_cell_magic('writefile', 'score.py', "import json\nimport numpy as np\nimport os\nimport pickle\nfrom sklearn.externals import joblib\nfrom sklearn.linear_model import LogisticRegression\n\nfrom azureml.core.model import Model\n\ndef init():\n    global model\n    # retreive the path to the model file using the model name\n    model_path = Model.get_model_path('Model_hurrthreat')\n    model = joblib.load(model_path)\n\ndef run(raw_data):\n    data = np.array(json.loads(raw_data)['data'])\n    # make prediction\n    y_hat = model.predict(data)\n    return json.dumps(y_hat.tolist())")


# In[28]:


from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies()
myenv.add_conda_package("scikit-learn")

with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())
with open("myenv.yml","r") as f:
    print(f.read())


# In[88]:


from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=1, 
                                               tags={"data": "MNIST",  "method" : "sklearn"}, 
                                               description='model hurrthreat',
                                               auth_enabled=1
                                               )


# In[90]:


get_ipython().run_cell_magic('time', '', 'from azureml.core.webservice import Webservice\nfrom azureml.core.image import ContainerImage\n\n# configure the image\nimage_config = ContainerImage.image_configuration(execution_script="score.py", \n                                                  runtime="python", \n                                                  conda_file="myenv.yml")\n\nservice = Webservice.deploy_from_model(workspace=ws,\n                                       name=\'sklearn-irene-svc\',\n                                       deployment_config=aciconfig,\n                                       models=[model],\n                                       image_config=image_config\n                                        )\n\nservice.wait_for_deployment(show_output=True)')


# In[94]:


print(service.scoring_uri)
print(list(service.get_keys()))


# In[66]:


import json
arr=np.array([5.46,63.35,1015.2,989,2.9])
arr=arr.reshape(1,-1)
print(arr.tolist())
test_samples = json.dumps({'data':arr.tolist()})
test_samples = bytes(test_samples, encoding = 'utf8')
result = service.run(input_data=test_samples)
print(result)


# In[105]:


#random_index = np.random.randint(0, len(X_test)-1)
import requests
import json
arr=np.array([[5.46,63.35,1015.2,989,2.9],[5.46,63.35,1015.2,989,2.9]])

input_data = "{\"data\":"+str(arr.tolist())+"}"
print(input_data)

headers = {'Content-Type':'application/json','Authorization':('Bearer CICX69Bi4Ctz4bQzHu1Tb1OdieVL9J7d')}

# for AKS deployment you'd need to the service key in the header as well
# api_key = service.get_key()
# headers = {'Content-Type':'application/json',  'Authorization':('Bearer '+ api_key)} 


resp = requests.(service.scoring_uri, input_data, headers=headers)

print("POST to url", service.scoring_uri)
#print("input data:", input_data)
#print("label:", y_test[random_index])
print("prediction:", resp.text)


# In[89]:


service.delete()


# In[ ]:




