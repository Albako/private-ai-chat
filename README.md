# Private and local AI assistant 

This project let's you host an AI Chat privately, without a subscription or giving up your or your company' precious data to other companies.

## Environment

This project will work only inside `Linux` environment. If you wanna use it inside `Windows`, then you have to open it inside `WSL2`.
All of this is created currently for `NVIDIA` GPU only. Without `NVIDIA` GPU the app won't work.

### Setting up the environment

You need Docker (Docker Desktop for Windows have to be configured in order to work inside `WSL2` environment) and NVIDIA Container-Toolkit.

## Starting the app

If you're starting the app for the first time (or if there were any changes made to the project), you have to use this commend:
```
docker compose up --build -d
```
If you're starting this app once again, then you can use only this
```
docker compose up -d
```

#### localhost

You can enter the web-gui using this link:
```
http://localhost:8080
```

## Shuting down the app

In order to shut down the app, you can use:
```
docker compose down -v
```
```
docker compose down
```
