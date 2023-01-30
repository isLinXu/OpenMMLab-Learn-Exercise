#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import pypandoc 

"""
　　pypandoc.convert_file(source_file, to, format=None, extra_args=(), encoding='utf-8',
                 outputfile=None, filters=None, verify_format=True)
    参数说明：
    source_file:源文件路径
    to：输入应转换为的格式；可以是'pypandoc.get_pandoc_formats（）[1]`之一
    format：输入的格式；将从具有已知文件扩展名的源_文件推断；可以是“pypandoc.get_pandoc_formats（）[1]”之一（默认值= None)
    extra_args：要传递给pandoc的额外参数（字符串列表）(Default value = ())
    encoding：文件或输入字节的编码 (Default value = 'utf-8')
    outputfile：转换后的内容输出路径+文件名，文件名的后缀要和to的一致，如果没有，则返回转换后的内容（默认值= None)
    filters – pandoc过滤器，例如过滤器=['pandoc-citeproc']
    verify_format：是否对给定的格式参数进行验证，（pypandoc根据文件名截取后缀格式，与用户输入的format进行比对）
     
    pypandoc.convert_text(source, to, format, extra_args=(), encoding='utf-8',
                     outputfile=None, filters=None, verify_format=True):
    参数说明：
    source：字符串       
    其余和canvert_file()相同      
 
"""

# With an input file: it will infer the input format from the filename
output = pypandoc.convert_file('../index.md', 'html', outputfile='../index.html')
