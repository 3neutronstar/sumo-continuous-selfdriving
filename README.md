# sumo_continuous-selfdriving
Reinforcement Learning based autonomous driving system(Continuous and Discrete combined)


### Contributor
Minsoo Kang<br/>
일규 Lee<br/>
Yunsik Cho<br/>

### How to use
  # choose mode
  simulate 대신 train, test 사용가능
  ```shell script
      python ./RL_main.py simulate
  ```
  #choose algorithm
  앞으로 구현될 알고리즘 선택가능(아직은 알고리즘 미구현)
  ```shell script
      python ./RL_main.py --algorithm alg
  ```
  #choose display
  gui로 볼 것인지 아닌지 선택가능 (필수적인지는 모르겠음)
  ```shell script
      python ./RL_main.py  --display True
  ```
### main.py 동작설명 및 목표
  
   configs.py에 있는 변수에 값을 상황에 따라 저장하고 불러오는 식으로
   파라미터들을 사용할 것
   
   main.py 실행시 선택option으로 잘돌아가게 하기 위해서는 연결시킬 AGENT, NET을
   만드는 클래스와의 연결이 잘되야할듯.
   
### 추가 수정(앞으로 계속 추가하고 고쳐나가면 좋을 것들)
  코드가 돌아가는 것 원리가 더 익숙해지면, 최대한 겹치는 코드 없이 간결하게 하기
  넘기는 변수의 수 줄여보기
  옵션을 추가해서 편해지면 추가하기
    
  
