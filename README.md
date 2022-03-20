# comparison_of_dnn
This repository is for getting better understainding of DNN. For now, I explored about DNN on california dataset and iris dataset. I'll explore it more with other dataset.

## Experiment results
Please see the following articles on Dev.to.

- [[Toward understanding DNN (deep neural network) well: California housing dataset](https://dev.to/ksk0629/toward-understanding-dnn-deep-neural-network-well-california-housing-dataset-3jp3)]
- [[Toward understanding DNN (deep neural network) well: iris dataset](https://dev.to/ksk0629/toward-understanding-dnn-deep-neural-network-well-iris-dataset-5179)]

I would appreciate it if someone shares tips or ccomments in the discussion box or this repository issues.

## Quickstart
### Preparation
1. Creating accounts

I'm quite sure someone who is seeing this repository has already got the account on git. Except for the account, we have got to create accounts on google and ngrok to start this quickstart.

2. Getting a personal access token and an authentication token

The first one is from git. We could see how to create it on the official page [[Creating a personal access token]](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token). Another token is found on a ngrok top page, like the following line.
```
$./grok authtoken [YOUR_TOKEN]
```

3. Uploading a config file to a `config` folder on a google drive

First, we have to create a `config` folder on google drive and then create and upload a config file. The config file should be named `general_config.ymal` and constructed like the following lines.

```yaml
github:
  username: your_username
  email: your_email@gmail.com
  token: your_personal_access_token
ngrok:
  token: ngrok_authentication_token
```

### Performing experiment
1. Cloning this repository on a local machine

2. Uploading `comparison_of_dnn.ipynb` to a google drive

3. Running all cells

After the cell in Run MLflow section was run, we could see the results of experiments on the outputted URL, like `MLflow Tracking UI: https://xxx-xx-xxx-xxx-xx.ngrok.io.`.
