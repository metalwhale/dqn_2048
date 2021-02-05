# 2048-dqn

## 1. Introduction
- Book: [Deep Reinforcement Learning Hands-On](https://www.packtpub.com/big-data-and-business-intelligence/deep-reinforcement-learning-hands)
- Gameplay: [Wiki](https://en.wikipedia.org/wiki/2048_(video_game)#Gameplay)
- Helpful repository: [DQN-2048](https://github.com/SergioIommi/DQN-2048)

## 2. Running
- Create `.env` file then edit its content:
  ```
  host$ cd infra/
  host$ cp .env.example .env
  ```
- Start container:
  ```
  host$ docker-compose up -d
  ```
- Run the code:
  <pre>
  host$ docker-compose exec dqn_2048 bash
  container$ python main.py <b>gpu_id</b>
  </pre>
