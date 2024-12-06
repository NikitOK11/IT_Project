# "Нарисуй идеальный кружочек" в новом свете! :cyclone:
## 📜Краткое описание
_____
Играли когда-нибудь в своем браузере в игру, в которой нужно вокруг центра мышкой нарисовать идеальный кружочек? 
Здесь я готов представить вам эту игру в новом свете! Запускайте файл проекта, ваша главная цель - нарисовать идеальный кружочек в __воздухе__, а программа 
_отследит_ движение пальца и _оценит_, насколько идеальный у вас рисунок!

## :chart_with_upwards_trend: Процесс игры
____
После запуска программы игру можно разделить на несколько этапов:
1. Начальная фаза - вам нужно будет разместить указательный палец перед камерой так, чтобы рука ___полностью помещалась в открывшемся окне___.
2. Фаза рисования - вам нужно рисовать кружочек __указательным пальцем__ в воздухе, программа будет фиксировать движение пальца и выводить на экран получившуюся фигуру. __ВАЖНО__: в процессе рисования ладонь должна полностью помещаться в поле зрения камеры, иначе программа может выдасть ошибку.
3. Конечная фаза - на экране показывается ваш результат и, в случае улучшения рекорда :trophy:, поздравления с его побитием :confetti_ball: Для запуска игры заново нужно нажать Enter.
## 🗂Структура проекта
____
Ключевые части проекта:
+ `project_file.py` - файл с основным скриптом игры.
+ `Settings` - файл со всеми переменными, которые используются в основном скрипте.
+ `results.txt` - файл, в котором хранится точность каждого нарисованного кружочка.


## Краткий гайд по использованию
____
Основные советы по использованию игры:
+ Всегда держите ладонь полностью в кадре
+ После того, как замкнете круг при рисовании, подождите чуть меньше секунды
