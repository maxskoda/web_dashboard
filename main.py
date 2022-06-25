from requests_html import HTMLSession, AsyncHTMLSession
import re

def get_values(session, str1=None, str2=None,):
    #create the session
    # session = HTMLSession()

    #define our URL
    # url = 'https://www.youtube.com/channel/UC8tgRQ7DOzAbn9L7zDL8mLg/videos'
    url='http://dataweb.isis.rl.ac.uk/IbexDataweb/default.html?Instrument=inter'
    #use the session to get the data
    r = session.get(url)

    #Render the page, up the number on scrolldown to page down multiple times on a page
    r.html.render()#sleep=1.2, keep_page=True, scrolldown=0)


    # #take the rendered html and find the element that we are interested in
    inst_pvs = r.html.find('#inst_pvs', first=True).text
    main = r.html.find('#groups', first=True).text

    if str1 is not None and str2 is not None:
        m_inst = re.search('(?<=' + str1 + ').*(?=' + str2 + ')', inst_pvs)
        m_main = re.search('(?<=' + str1 + ').*(?=' + str2 + ')', main)
        return m_inst.group(0)
    else:
        values = {}
        m = re.search('(?<=RB Number:).*(?=:)', inst_pvs)
        try:
            values['rbno'] = m.group(0)
        except AttributeError:
            values['rbno'] = "N/A"

        m = re.search('(?<=Run Status:).*(?=:)', inst_pvs)
        try:
            values['run_status'] = m.group(0)
        except AttributeError:
            values['run_status'] = "N/A"

        m = re.search('(?<=Run Number:).*(?=:)', inst_pvs)
        try:
            values['run_number'] = m.group(0)
        except AttributeError:
            values['run_number'] = "N/A"

        m = re.search('(?<=Title:\s\s).*(?=:)', inst_pvs)
        try:
            values['title'] = m.group(0)
        except AttributeError:
            values['title'] = "N/A"

        m = re.search('(?<=PHI:\s\s).*(?=deg:)', main)
        try:
            values['PHI'] = m.group(0)
        except AttributeError:
            values['PHI'] = "N/A"

        m = re.search('(?<=THETA:\s\s).*(?=deg:)', main)
        try:
            values['THETA'] = m.group(0)
        except AttributeError:
            values['THETA'] = "N/A"

        m = re.search('(?<=S1VG:\s\s).*(?=:)', main)
        try:
            values['S1VG'] = m.group(0)
        except AttributeError:
            values['S1VG'] = "N/A"

        m = re.search('(?<=S1HG:\s\s).*(?=:)', main)
        try:
            values['S1HG'] = m.group(0)
        except AttributeError:
            values['S1HG'] = "N/A"

        m = re.search('(?<=S2VG:\s\s).*(?=:)', main)
        try:
            values['S2VG'] = m.group(0)
        except AttributeError:
            values['S2VG'] = "N/A"

        m = re.search('(?<=S2HG:\s\s).*(?=:)', main)
        try:
            values['S2HG'] = m.group(0)
        except AttributeError:
            values['S2HG'] = "N/A"

        m = re.search('(?<=SM1ANGLE:\s\s).*(?=deg:)', main)
        try:
            values['SM1ANGLE'] = m.group(0)
        except AttributeError:
            values['SM1ANGLE'] = "N/A"

        m = re.search('(?<=SM1INBEAM:\s\s).*(?=:)', main)
        try:
            values['SM1INBEAM'] = m.group(0)
        except AttributeError:
            values['SM1INBEAM'] = "N/A"

        m = re.search('(?<=SM2ANGLE:\s\s).*(?=deg:)', main)
        try:
            values['SM2ANGLE'] = m.group(0)
        except AttributeError:
            values['SM2ANGLE'] = "N/A"

        m = re.search('(?<=SM2INBEAM:\s\s).*(?=:)', main)
        try:
            values['SM2INBEAM'] = m.group(0)
        except AttributeError:
            values['SM2INBEAM'] = "N/A"

        m = re.search('(?<=TRANS:\s\s).*(?=:)', main)
        try:
            values['TRANS'] = m.group(0)
        except AttributeError:
            values['TRANS'] = "N/A"

        m = re.search('(?<=HEIGHT:\s\s).*(?=:)', main)
        try:
            values['HEIGHT'] = m.group(0)
        except AttributeError:
            values['HEIGHT'] = "N/A"

        m = re.search('(?<=HEIGHT2:\s\s).*(?=:)', main)
        try:
            values['HEIGHT2'] = m.group(0)
        except AttributeError:
            values['HEIGHT2'] = "N/A"

        return values
