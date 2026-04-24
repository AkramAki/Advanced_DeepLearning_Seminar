.PHONY: all new plots report clean

TASK ?= task_template

all:
	$(MAKE) -C tasks/$(TASK)

new:
	cp -r tasks/task_template tasks/$(NAME)

plots:
	$(MAKE) -C tasks/$(TASK) plots

report:
	$(MAKE) -C tasks/$(TASK) report

clean:
	$(MAKE) -C tasks/$(TASK) clean