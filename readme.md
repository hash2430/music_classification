* 프로그램 사용 방법:
1. NotRunnables/Path.py에 method1_path, method2_path, method3_path, data_path를 각각 적습니다.
method1은 learning 방법을, method2는 feature exteraction, method3은 feature summary을 다양하게 해보는 approach입니다.
이전 approach에서 최고의 성능을 낸 방법이 다음 approach에 채택됩니다.
예를 들면 method1에서 다양한 러닝 알고리즘을 시도해보니 rbf kernel의 nonlinear svm이 가장 성능이 좋아서 method2, 3에서의
러닝 알고리즘은 모두 rbf kernel의 nonlinear svm으로 픽스되었고 feature와 feature summary 방법이 달라집니다.

2. Runnables/Main.py를 run합니다.
3. 각 methodn_path/report 폴더에 실험별 리포트 파일이 떨어진 것을 확인하면 training time, validation accuracy,
 hyper paramenters, test accuracy 등을 확인 할 수 있습니다.

* 패키지 구성:
1. Baseline: 과제에 포함돼 있던 baseline 코드입니다.
2. NotRunnables: 실험과정에서 반복적으로 사용되는 함수들, 클래스들, 모듈들을 모아놓은 패키지입니다.
3. Runnables: 실행파일인 Main.py가 포함되어 있습니다.

* 폴더 구성:
Main.py를 돌리면 Path.py에 설정한 각 method 폴더마다 하위에 extracted feature와 summary feature의 폴더가 생깁니다.
먼저 실행된 실험에서 추출된 feature나 summary된 feature를 재사용하는 경우에는 이를 참조하며, 중복되는 작업을 재수행하지는 않습니다.
예를 들어 method3에서는 feature summary를 다양하게 시도하는 접근이기 때문에 이미 method2에서 생성된 60 dimension의 mfcc를 재사용해
summarization만 여러 방법으로 합니다.

* 프로그램 설계 의도:
중간 과정들을 분석하기 쉽도록 모든 단계마다 아웃풋 파일로 기록하도록 프로그래밍하였습니다.
파일 i/o가 늘어 수행시간은 베이스라인보다 길 수 있습니다.
하지만 성능 파일에 기록할 시간은 실제 알고리즘이 수행된 시간이기 때문에 파일 i/o 시간은 제외하였습니다.

설계 시에 중요하게 생각한 것은 프로그램의 수행 시간을 희생해 개발자(본인)의 혼돈을 최소화하고
다양한 조합의 실험이 가능하도록 하였습니다.

* 본인의 깃에 올려두었습니다.
설계까지는 푸쉬하였지만 평가가 이루어지기 전까지 실험 내용은 푸쉬하지 않았습니다.
https://github.com/hash2430/music_classification
