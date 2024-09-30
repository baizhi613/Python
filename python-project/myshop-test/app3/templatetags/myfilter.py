from django import template

register = template.Library()
@register.filter
def show_title(value,n):
    if len(value) > n:
        return f'{value[0:n]}...'
    else:
        return value