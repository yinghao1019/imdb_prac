<h1 aling="left">Imdb_prac</h1>
<p >Imdb_prac is a nlp ai project tha can use google cloud Vertex AI to training.</p>

## **Project Articture**
### **Main part & flow**
 ![Repo](https://i.imgur.com/3n6ImYg.jpg)

* Generate : Prepare clean data in csv format & trained tokenizer before training Model.If you don't have any file, this will be required.  

* Train Pipe : The training model's core part.Contain Normalize(remove conse、html、punctucation,lower case)  
                ,Preprocess(NER,BPE encoding),Postprocess(Padding sequence,convert idx).  

* Cloud Storage : This will used for store trained model,tokenizer,data,deploy model to serve prediction. 

### **Module & Feature Articture**
```bash
|
|
+---imdb_prac
|   |   settings.py
|   |   task.py
|   |   utils.py
|   |
|   +---models
|   |   |   hsrnn.py
|   |   
|   +---process
|   |   |   data.py
|   |   |   generate.py
|   |   |   text.py
|   |   
|   +---trainer
|   |   |   pipeline.py
|   |   
|   |   

```
* models : The model articture which use rnn,inner-attention,softmax.

* process : 
    * Data : used to build training model dataset,iterator with pytorch format.
    * text : Contain process text function,like remove html tag,extract & tokenize text.
    * generate : Used to train tokenizer & clean rwa file to csv.

* trainer : pipeline is training model flow,include Amp training,learning rate schedular,Gradient Normalizer.

* Other :
    * task : Training model entrypoint.You can use optional arg to adjust training model flow.
    * utils/settings : Extract specific env values,Model metrics func,and project log settings.


### **Model Articture**
![Model](https://i.imgur.com/Ktk23tj.jpg)

* Main articture :We use two level encoder to extract feature.So in the first step,must split whole text to each sent.In addition,also use NER tagger to extract linguistic feature.And we concat it over embed feature.At sent level encoder,we use it to extract each sent context feature,and in Document level encoder,we convert seq of sent context to one vector for after fully connected layer.

* Encoder block : use to stack rnn layer to extract and combine it into context.

Refernce : The model design concept is reference to https://arxiv.org/abs/1811.00706.


### **Built with**
* pytorch  
* GCP storage api  
* spacy  

<h1 aling="left">Getting started</h1>

#### **Prerequisites**
* IMDB sentiment dataset
* Google cloud sdk
* pytorch
#### **Installation**

1. clone this repository

   ```bash
   git clone https://github.com/yinghao1019/imdb_prac.git
   ```
2. download this dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and move to project dir
3. Create bucket on your cloud project
4. set ENV for training
   ```CMD
   set BUCKET_NAME=${your storage bucket}
   set PROJECT_ID=${project id}
   set CLOUD_TRAINDATA_PATH=gs//:${your train data path}
   set CLOUD_TESTDATA_PATH=gs//:${your test data path}
   set CLOUD_TOKENIZER_PATH=gs//:${your tokenizer data path}
   set AIP_MODEL_DIR=gs://${your model path}
   set PYTHONPATH=.
   ```

### **Usage**
1.  Before use training application.Confirm you have clean csv data & tokenizer.  
    Please clean data and run below command if you don't.  

    ```bash 
    python generate.py data ${your_data_path} ${cloud_data_path} -n 12500  
    ```

    And then useing clean data to train tokenizer

    ```bash 
    python generate.py token ${your_data_path} ${cloud_token_path} -ms 25000
    ```

2.  Run the main program to training NLP model

    ```bash 
    python task.py  --ep 10 --warm_ep 3 --batch_size 64 --max_sentN 5
    ```

## **Start with Vertex Ai**
If you training models's resource is limited,it's  highly recommended that you can use the GCP   
service-Vertex ai to create custom training.PLease click this link to know more :  
[Vertex AI custom training documentation.](https://cloud.google.com/vertex-ai/docs/training/custom-training)

1. You should use docker to buid container for custom training,So must run below command:  
    ```bash
    docker build -t={your_image_name_in_Artifact_registry_Repo} .
    ```

2. Before let Vertex Ai to use your customized container,you should create repository in Artifact registry.  
    Please read [Artifact Registry documentation](https://cloud.google.com/artifact-registry/docs/manage-repos) to create.  

3. Then set up your authentication to allow docker access cloud service.  
    Please read [Work with container Image](https://cloud.google.com/artifact-registry/docs/docker) to know how to use.

## **Learn More**
After you training ,if you want to serve for other application,please follow below link.
[Deploy trained model service with Vertex Ai](https://github.com/yinghao1019/imdb_infer)

## **Contact**
Ying Hao Hung-1104137203@gmail.com
Project link : https://github.com/yinghao1019/imdb_prac



