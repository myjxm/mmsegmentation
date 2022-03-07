#!flask/bin/python
from flask import Flask,jsonify,request
import os
import json
import datetime
import subprocess
import pymysql
import logging
import time
import threading
import requests
import _thread
from mysql_helper import *
from impala.dbapi import connect
from impala.util import as_pandas
app = Flask(__name__)


sys.path.append("/home/hdwetl/script/shared")

from mysql_helper import *
app = Flask(__name__)


res=[
    {"code":0,"msg" :'create table success!'},
    {"code":1,"msg" : 'can not get the table list!'},
    {"code": 1,"msg": 'failed to create table!'},
    {"code":0,"msg" :'create user success!'},
    {"code": 1,"msg": 'failed to create group!'},
    {"code": 1, "msg": 'failed to create user!'},
    {"code":0,"msg" :'grant role success!'},
    {"code":0,"msg" :'succeed to change password!'},
    {"code":1,"msg" :'failed to change password!'}
]
ctl_data_conn_conf = {'user':'root','password':'Abcd#2021','host':'55.14.15.156','database':'ctl_data'};
metastore_conn_conf = {'user':'root','password':'password','host':'55.14.15.156','database':'metastore'};
ctl_data_conn_conf = {'user':'root','password':'password','host':'55.14.15.156','database':'ctl_data'};
metastore_conn_conf = {'user':'root','password':'password','host':'55.14.15.156','database':'metastore'};
impala_host = 'hdwdexcli1'
impala_port = 25004
hive_kerberos_tgt = "export KRB5CCNAME='/tmp/krb5cc_0'"
IMPALA_CONN_CONF = {'host':'hdwcdpvm04ts','port':21050}

#生产IP：
#URL_IP = "12.0.229.65"
#测试IP：
URL_IP = "55.14.103.110"
URL = r"http://" + URL_IP + r":5000/dvt/data/check"

#txdate = sys.argv[1]
#folder = sys.argv[2]
#file_name = sys.argv[3]
def impala_query_dvt(sql_cmd,conn_conf=IMPALA_CONN_CONF,user=None):
    if user is not None:
       kbrs_init(user)
    result=()
    exeRes = ''
    host_v = conn_conf['host']
    port_v = conn_conf['port']
    try:
        conn = connect(host=host_v, port=port_v, auth_mechanism='GSSAPI',kerberos_service_name='impala')
    except Exception as e:
        print("connect fails!:{}".format(e))
        return "error:impala connect error " + str(e)
    try:
        cur = conn.cursor()
        cur.execute(sql_cmd)
        result = as_pandas(cur).astype('str').values.tolist()
        if len(result) != 0:
            exeRes = str(result[0])
    except Exception as e:
        print("<!--:PRINT{O_RTNCOD:-1,O_RTNMSG:'" + 'connect fails!{}'.format(e) + "'}-->")
        exeRes = "error:sql execute error " + str(e)
        updateRes = mysql_update("update ctl_data.m15_hdm_dvt_query_res set res = \"" + exeRes + "\" where impalasql = '" + sql_cmd + "' order by upd_dt desc limit 1")
        if updateRes['code'] == 0 :
            logging.info('update ctl_data.m15_hdm_dvt_query_res success!')
        else :
            logging.error(updateRes['message'])
        return "error:sql execute error " + str(e)
    finally:
        cur.close()
        conn.commit()
        conn.close()
    #print(type(result[0]))   list
    updateRes = mysql_update("update ctl_data.m15_hdm_dvt_query_res set res = \"" + exeRes + "\" where impalasql = '" + sql_cmd + "' order by upd_dt desc limit 1")
    if updateRes['code'] == 0 :
        logging.info('update ctl_data.m15_hdm_dvt_query_res success!')
    else :
        logging.error(updateRes['message'])
    return result




# def mysql_query(sql_cmd,params,conn_conf):
    # result=();
    # try:
        # print(conn_conf);
        # conn = pymysql.connect(**conn_conf )
    # except pymysql.Error as e:
        # print('connect fails!{}'.format(e))
        # print("<!--:PRINT{O_RTNCOD:-1,O_RTNMSG:'" + 'connect fails!{}'.format(e) + "'}-->")
        # return "error:mysql connect error " + str(e)
    # cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)

    # try:
        # if len(params)>0 :
           # for param in params:
               # cursor.execute(sql_cmd,param)
        # else :
               # cursor.execute(sql_cmd)
               # result = cursor.fetchall()
    # except pymysql.Error as e:
        # print("<!--:PRINT{O_RTNCOD:-1,O_RTNMSG:'" + 'sql execute fails!{}'.format(e) + "'}-->")
        # return "error:sql execute error " + str(e)
    # finally:
        # cursor.close()
        # conn.commit()
        # conn.close()
    # return result;

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/update/create_table_status/<status>', methods=['GET'])
def update_create_table_status(status):
        sql = "insert into m15_hdm_current_init_state(id,step1,step2,step3,step4,step5,`current_date`,ins_tm,upd_tm) values(replace(uuid(),'-',''),'Y','" + status + "','D','D','D',current_date(),current_timestamp(),current_timestamp()) ON DUPLICATE KEY UPDATE step1 = VALUES(step1),step2 = VALUES(step2),step3 = VALUES(step3),step4 = VALUES(step4),step5 = VALUES(step5),upd_Tm = VALUES(upd_Tm);" 
        updres = mysql_query(sql, [], ctl_data_conn_conf)
        print(updres)
        if len(updres) == 0:
           return jsonify(res[0])
        else:
           return jsonify(res[2]) 

@app.route('/create_table/hdw', methods=['POST','GET'])
def hdw_table_create():
        #time.sleep(3600)
        data = request.stream.read()
        print(data.decode())
        #a = 'nohup perl  /home/hdwetl/script/shared/init/hdw_object_translate.pl.json  ' + data.decode() + ' &'
        #a = 'nohup python3 /home/hdwetl/script/shared/createTable.py ' + datetime.datetime.now().strftime("%Y%m%d") + ' ctl_data' + ' createTable ' +  data.decode() + ' &'
        #a = 'nohup python3 /home/hdwetl/script/shared/createTable.py ' + datetime.datetime.now().strftime("%Y%m%d") + ' ctl_data' + ' createTable '+ ' &'
        #print(a)
        cmd = 'nohup python3 /home/hdwetl/script/shared/createTable.py ' + datetime.datetime.now().strftime("%Y%m%d") + ' ctl_data' + ' createTable '+ data.decode()  + ' &'
        p = subprocess.Popen(cmd, shell = True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        output,err = p.communicate()
        #out=os.system(a)
        logging.info('output:'+str(output))
        logging.info('err:'+str(err))
        out = str(output)
        currLen = len(out) - 2
        result = {}
        result['code'] = out[1]
        result['msg'] = out[3:currLen]
        return json.dumps(result)
        #else:
        #   return jsonify(res[2])

#@app.route('/create_table/hdw', methods=['POST','GET'])
#def hdw_table_create():
        #data = request.stream.read()
        #a = "perl  /home/hdwetl/script/shared/init/hdw_object_translate.pl.json  " + data.decode()
        #print(a)
        #out=os.system(a)
        #print(out)
        #print('success create table ')
#        out=0
#        if out == 0:
#           print(jsonify(res[0]))
#           return jsonify(res[0])
#        else:
#           print(jsonify(res[2:]))
#           return jsonify(res[2])
@app.route('/query/cdh_impala_query', methods=['POST','GET'])
def cdh_impala_query():
     kbrs_init('dvt_query')
     data = request.stream.read()
     data = data.decode()
     print(data)
     data = json.loads(data)
     print(data["sql"])
     #data = '{\"sql\":\"select * from pdw_tdw.dvt_test_lpf002 limit 1;\"}'
     #kbrs_init('etl_user@CMBCHINA.COM')
     #sql = 'select * from pdw_tdw.dvt_test_lpf002 limit 1;'
     #logging.info('sql:'+ str(sql))
     # res = impala_query_dvt(data["sql"],IMPALA_CONN_CONF,'etl_user@CMBCHINA.COM')
     #res = impala_query_dvt(sql,IMPALA_CONN_CONF,'etl_user@CMBCHINA.COM')
     #res = impala_query_dvt(sql)
     #logging.info('res:'+ str(res))
     try :
         updateRes = mysql_update("insert into ctl_data.m15_hdm_dvt_query_res (`impalasql`) values ('" + data["sql"] + "')" )
         res = mysql_query_new("select * from ctl_data.m15_hdm_dvt_query_res where impalasql = '" + data["sql"] + "' order by upd_dt desc limit 1")
         if res['code'] == 0:
             queryid = res['data'][0]['id']
         else :
             logging.error(res['message'])
         _thread.start_new_thread(impala_query_dvt,(data["sql"],IMPALA_CONN_CONF,'etl_user@CMBCHINA.COM',))
         return  json.dumps({"code":0,"msg":"The query request has been received!","query_id":queryid})
     except :
         print("error : 提交任务失败！")
         return json.dumps({"code":-1,"msg":"No query request was received!","query_id":"null"})
     # if str(res).startswith('error'):
        # return  json.dumps({"code":1,"msg":str(res)})
     # else : 
        # return  json.dumps({"code":0,"msg":res})

@app.route('/query/cdh_impala_query_id/<query_id>', methods=['GET'])
def cdh_impala_query_id(query_id):
    try :
        res = mysql_query_new("select * from ctl_data.m15_hdm_dvt_query_res where id = " + query_id + " order by upd_dt desc limit 1")
        if res['code'] == 0:
            queryRes = res['data'][0]['res']
            if queryRes is None :
                return  json.dumps({"code":0,"msg":"The data is being queried...","data":"null"})
            else :
                if "error:sql execute error " in queryRes :
                    return  json.dumps({"code":-1,"msg":str(queryRes),"data":"null"})
                else :
                    return  json.dumps({"code":0,"msg":"Data query succeeded !","data":str(queryRes)})
        else :
            logging.error(res['message'])
            return  json.dumps({"code":-1,"msg":res['message'],"data":"null"})
    except :
        return json.dumps({"code":-1,"msg":"Task submission failure","data":"null"})

@app.route('/query/dvt_query_job_cfg', methods=['post','GET'])
def dvt_query_job_cfg():
    data = request.stream.read()
    data = data.decode()
    data = json.loads(data)
    hdw_db_nm = data['hdw_db_nm']
    hdw_tbl_nm = data['hdw_tbl_nm']
    res = mysql_query_new("select hdw_db_nm,hdw_tbl_nm,tbl_form,splt_fld from ctl_data.m15_hdm_job_cfg where hdw_db_nm = lower('" + hdw_db_nm + "') and hdw_tbl_nm = lower('" + hdw_tbl_nm + "')")
    print(res)
    if res['code'] == 0 and len(res['data']) > 0 :
        queryRes = res['data'][0]
        return json.dumps({"code":0,"msg":str(queryRes)})
    elif res['code'] != 0 and len(res['data']) > 0  :
        return json.dumps({"code":-1,"msg":res['message']})
    else :
        return json.dumps({"code":-1,"msg":"sql error"})

@app.route('/query/dvt_check', methods=['post','GET'])
def dvt_check():
    data = request.stream.read()
    data = data.decode()
    data = json.loads(data)
    # req_id = datetime.datetime.now().strftime(format="%Y%m%d%H%M%S%f")
    # res = impala_query("select src_service,tbl_eng_nm,dat_slice_val from ctl_data.m16_dlm_slice_sts_inf_kudu where dat_slice_sts = 'I' and tbl_eng_nm = '" + tbl_eng_nm + "'")
    # if res['code'] == 0 and len(res['data']) > 0 :
        # for i in res['data'] :
            # src_service = i[0]
            # tbl_eng_nm = i[1]
            # minSpltFld = i[2]
            # maxSpltFld = i[2]
    req_id = data['req_id']
    req_src = data['req_src']
    tbl_eng_nm = data['tb']
    plf = str(data['plf'])
    m_plf = data['m_plf']
    splt_st_dt = data['splt_st_dt']
    splt_ed_dt = data['splt_ed_dt']
    dw_dat_dt = time.strftime('%Y-%m-%d',time.localtime(time.time()))
    # type(data['minSpltFld'])
    # dvtData = {}
    # dvtData['req_id'] = req_id
    # dvtData['req_src'] = 'hdw'
    # dvtData['plf'] = [src_service,'hdw']
    # dvtData['typ'] = 'tb_normal'
    # dvtData['tb'] = tbl_eng_nm
    # dvtData['splt_st_dt'] = minSpltFld
    # dvtData['splt_ed_dt'] = maxSpltFld
    # dvtData['m_plf'] = src_service
    return_data = requests.post(URL,json=data)
    res = json.loads(return_data.text)
    print(res)
    if(res['rs']):
        logging.error("Submit dvtData check failed...")
        return jsonify({"code":-1,"msg":"Submit dvtData check failed..."})
    else:
        print("insert into ctl_data.m16_kudu_dvt_check_res(`req_id`,`req_src`,`tbl_eng_nm`,`splt_st_dt`,`splt_ed_dt`,`check_res`,`plf`,`m_plf`,`dw_dat_dt`)values('" + req_id + "','" + req_src + "','" + tbl_eng_nm + "','" + splt_st_dt + "','" + splt_ed_dt + "','R',\"" + plf + "\",'" + m_plf + "','" + dw_dat_dt + "')")
        updateRes = mysql_update("insert into ctl_data.m16_kudu_dvt_check_res(`req_id`,`req_src`,`tbl_eng_nm`,`splt_st_dt`,`splt_ed_dt`,`check_res`,`plf`,`m_plf`,`dw_dat_dt`)values('" + req_id + "','" + req_src + "','" + tbl_eng_nm + "','" + splt_st_dt + "','" + splt_ed_dt + "','R',\"" + plf + "\",'" + m_plf + "','" + dw_dat_dt + "')")
        logging.info("Submit dvtData check successfully...")
        return jsonify({"code":0,"msg":"Submit dvtData check successfully..."})

@app.route('/create_table/tdd', methods=['POST','GET'])
def tdd_table_create():
       data = request.stream.read()
       print(data)
       a = 'perl /home/hdwetl/script/shared/init/init_cs.pl.json ' + data.decode()
       print(a)
       out=os.system(a)
       print(out)
       if out==0:
          return jsonify(res[0])
       else:
          return jsonify(res[2])
@app.route('/query/cdh_impala_assignments_rate', methods=['POST','GET'])
def cdh_impala_assignments_rate():
       current_date_time = datetime.datetime.now()
       last_min = current_date_time - datetime.timedelta(minutes=1) - datetime.timedelta(hours=9)
       current_min = current_date_time - datetime.timedelta(hours=9)
       _from = '&from=' + str(last_min.date()) + 'T' + str(last_min.hour) + '%3A' + str(last_min.minute) + '%3A' + str(
           last_min.second) + 'Z'
       curl_cmd= "curl -u admin:admin -X GET --header 'Accept: application/json' 'http://11.6.72.7:7180/api/v19/timeseries?contentType=application%2Fjson&desiredRollup=RAW" + _from + "&mustUseDesiredRollup=false&query=SELECT%20total_assignments_rate_across_impalads%20WHERE%20entityName%20%3D%20%22impala%22%20AND%20category%20%3D%20SERVICE&to=now'"
       impala_job_num = subprocess.check_output(curl_cmd,shell=True)
       str_impala_job_num = impala_job_num.decode('utf-8')
       str_impala_job_num=str_impala_job_num.replace("\\n", "")
       #json_impala_job_num = json.loads(str_impala_job_num)
       return str_impala_job_num
@app.route('/query/database_storage/all', methods=['POST','GET'])
def database_storage():
    hdw_space = mysql_query(
        "select Db_Nm, total_space, used_space,cast(log_tm as char) as log_tm  from ctl_data.m18_db_space_sts where Db_Nm = 'HDW' order by log_tm desc limit 1;",
        [], ctl_data_conn_conf)
    if isinstance(hdw_space,str) and  hdw_space[:5] == 'error':
        return hdw_space
    tdd_data_space = mysql_query(
        "select Db_Nm, total_space, used_space,cast(log_tm as char) as log_tm from ctl_data.m18_db_space_sts where Db_Nm = 'TDD_DATA' order by log_tm desc limit 1;",
        [], ctl_data_conn_conf)
    if isinstance(tdd_data_space,str) and  tdd_data_space[:5] == 'error':
        return tdd_data_space
    tdd_data_d_space = mysql_query(
        "select Db_Nm, total_space, used_space,cast(log_tm as char) as log_tm  from ctl_data.m18_db_space_sts where Db_Nm = 'TDD_DATA_D' order by log_tm desc limit 1;",
        [], ctl_data_conn_conf)
    if isinstance(tdd_data_d_space,str) and  tdd_data_d_space[:5] == 'error':
        return tdd_data_d_space
    tdd_data_a_space = mysql_query(
        "select Db_Nm, total_space, used_space,cast(log_tm as char) as log_tm  from ctl_data.m18_db_space_sts where Db_Nm = 'TDD_DATA_A' order by log_tm desc limit 1;",
        [], ctl_data_conn_conf)
    if isinstance(tdd_data_a_space,str) and  tdd_data_a_space[:5] == 'error':
        return tdd_data_a_space
    result = {}
    result['hdw'] = hdw_space
    result['tdd_data_space'] = tdd_data_space
    result['tdd_data_a_space'] = tdd_data_a_space
    result['tdd_data_d_space'] = tdd_data_d_space
    return jsonify(result)
@app.route('/query/database_table_top10/<database>/<ins_dt>', methods=['POST','GET'])
def database_table_top10(database,ins_dt):
    #database=database.encode('utf-8')
    top10 = mysql_query("select tbl_nm,tbl_size,cast(ins_dt as char) as ins_dt  from m18_td_tbl_top10 where ins_dt='" + ins_dt + "' and db_nm='" + str.upper(database.strip()) + "';",[], ctl_data_conn_conf)
    if isinstance(top10, str) and top10[:5] == 'error':
        return top10
    return jsonify(top10)
@app.route('/query/etl_progress/<type>/<etl_dt>', methods=['POST','GET'])
def etl_progress(type,etl_dt):
    #etl_dt = '2019-12-08'
    etl_dt = datetime.datetime.strptime(etl_dt,"%Y-%m-%d")
    etl_dt = etl_dt + datetime.timedelta(days=-1)
    etl_dt = etl_dt.strftime("%Y-%m-%d")
    avg_progress= mysql_query("select cast(avg(complete_num/total_num) as char) as avg,log_time_id from ctl_data.m18_tdd_etl_progress where opr_typ_cd = '" + type + "' group by log_time_id;",[], ctl_data_conn_conf)
    if isinstance(avg_progress, str) and avg_progress[:5] == 'error':
        return avg_progress
    current_progress = mysql_query("select complete_num,total_num,cast(complete_num/total_num as char) percentage,log_time_id from ctl_data.m18_tdd_etl_progress where opr_typ_cd = '" + type + "' and etl_dt = '" + etl_dt + "' group by log_time_id;",[], ctl_data_conn_conf)
    if isinstance(current_progress, str) and current_progress[:5] == 'error':
        return current_progress
    result={}
    result['avg_progress'] = avg_progress
    result['current_progress']=current_progress
    return jsonify(result)
@app.route('/query/hdw_comprehensive_quota/<quota>/<num>', methods=['POST', 'GET'])
def hdw_comprehensive_quota(quota,num):
    res = mysql_query("select  " + quota + ",cast(stat_dt as char) stat_dt  from ctl_data.m18_hdw_comprehensive_quota  order by stat_dt desc limit " + str(num) + ";", [], ctl_data_conn_conf)
    if isinstance(res, str) and res[:5] == 'error':
        return res
    return jsonify(res)
@app.route('/query/user_privilege', methods=['POST', 'GET'])
def user_privilege():
    data = request.stream.read()
    datadict = json.loads(data)
    result = []
    for db in datadict.get("selectTables"):
        for table in datadict.get("selectTables").get(db):
            is_exist = mysql_query("select b.name as db_name,a.TBL_NAME as tablename from TBLS a left join DBS b on a.DB_ID=b.DB_ID where a.TBL_NAME =  '" + table + "' and b.name =  '" + db + "';",[],metastore_conn_conf);
            if len(is_exist) == 0 :
                result.append(
                    {"privilegeUserName": datadict.get("privilegeUserName"), "dbName": db, "tableName": table,
                     "privileges": []})
                continue;
            res = mysql_query(
                "select b.ACTION from ctl_data.m17_team_role_info a inner join ctl_data.m17_group_role_privilege_info b on a.group_name=b.group_name where a.user_name = '" + datadict.get(
                    "privilegeUserName") + "' and b.db_name = '" + db + "' and b.table_name = '" + table + "';",
                [], ctl_data_conn_conf)
            if isinstance(res, str) and res[:5] == 'error':
                return res
            if len(res) == 0:
                res = mysql_query(
                    "select b.ACTION from ctl_data.m17_team_role_info a inner join ctl_data.m17_group_role_privilege_info b on a.group_name=b.group_name where a.user_name = '" + datadict.get(
                        "privilegeUserName") + "' and b.db_name = '" + db + "' and b.table_name = '__NULL__';",
                    [], ctl_data_conn_conf)
                if isinstance(res, str) and res[:5] == 'error':
                    return res
                if len(res) == 0:
                    res = mysql_query(
                        "select b.ACTION from ctl_data.m17_team_role_info a inner join ctl_data.m17_group_role_privilege_info b on a.group_name=b.group_name where a.user_name = '" + datadict.get(
                            "privilegeUserName") + "' and b.db_name = '__NULL__' and b.table_name = '__NULL__';",
                        [], ctl_data_conn_conf)
                    if isinstance(res, str) and res[:5] == 'error':
                        return res
                    if len(res) == 0:
                        result.append(
                            {"privilegeUserName": datadict.get("privilegeUserName"), "dbName": db, "tableName": table,
                             "privileges": []})
                    else :
                        result.append(
                            {"privilegeUserName": datadict.get("privilegeUserName"), "dbName": db, "tableName": table,
                             "privileges": [key.get("ACTION") for key in res]})
                else :
                    result.append(
                        {"privilegeUserName": datadict.get("privilegeUserName"), "dbName": db, "tableName": table,
                         "privileges": [key.get("ACTION") for key in res]})

            else:
                result.append({"privilegeUserName": datadict.get("privilegeUserName"), "dbName": db, "tableName": table,
                               "privileges": [key.get("ACTION") for key in res]})
    return jsonify(result)
@app.route('/create_user/<user>/<group>/<team>/<type>', methods=['POST', 'GET'])
def create_user(user,group,team,type):
    kbrs_init('hive')
    if type == 'I':
        is_group = mysql_query(
            "select group_name  from ctl_data.m17_group_role_privilege_info where group_name = 'gu_itc_" + group + "' limit 1;",
              ctl_data_conn_conf)
        if isinstance(is_group, str) and is_group[:5] == 'error':
            return is_group
        if len(is_group) == 0 :
            subprocess.check_output(hive_kerberos_tgt, shell=True)
            os.environ["KRB5CCNAME"] = hive_kerberos_tgt_path
            conn = connect(host=impala_host, port=impala_port, auth_mechanism='GSSAPI', kerberos_service_name='impala')
            cur = conn.cursor()
            cur.execute('create role rd_' + group + ';')
            cur.execute('create database if not exists  litc_' + team + ';')
            cur.execute('grant all on database litc_' + team + ' to  role rd_' + group + ';')
            create_group = "sh adduser.sh -g  gu_itc_" + group + " rd_" + group
            result = subprocess.call(create_group, shell=True)
            if result != 0 :
                return jsonify(res[4])
            else:
                result =mysql_query(
                    "insert into ctl_data.m17_group_role_privilege_info values('gu_itc_" + group + "','rd_" + group +  "','DB','server1','litc_" + team + "',null,null,null,'ALL')", ctl_data_conn_conf)
                if isinstance(result, str) and result[:5] == 'error':
                    return result
            create_user = "sh adduser.sh -u U" + user + " Cmb@6666 gr_itc,gu_itc_" + group + " Y"
            result = subprocess.call(create_user, shell=True)
            cur.close()
            conn.close()
            if result != 0 :
                return jsonify(res[5])
            else:
                result = mysql_query(
                    "replace into ctl_data.m17_team_role_info values('U" + user + "','gu_itc_" + group + "','litc_" + team + "','rd_" + group + "');", ctl_data_conn_conf)
                if isinstance(result, str) and result[:5] == 'error':
                    return result
                return jsonify(res[3])
        else:
            create_user = "sh adduser.sh -u U" + user + " Cmb@6666 gr_itc,gu_itc_" + group + " Y"
            result = subprocess.call(create_user, shell=True)
            if result != 0:
                return jsonify(res[5])
            else:
                result = mysql_query(
                    "replace into ctl_data.m17_team_role_info values('U" + user + "','gu_itc_" + group + "','litc_" + team + "','rd_" + group + "');",
                     ctl_data_conn_conf)
                if isinstance(result, str) and result[:5] == 'error':
                    return result
                return jsonify(res[3])
    else :
        is_group = mysql_query(
            "select group_name  from ctl_data.m17_group_role_privilege_info where group_name = 'gu_hqb_" + team + "' limit 1;",
            ctl_data_conn_conf)
        if isinstance(is_group, str) and is_group[:5] == 'error':
            return is_group
        if len(is_group) == 0:
            subprocess.check_output(hive_kerberos_tgt, shell=True)
            os.environ["KRB5CCNAME"] = hive_kerberos_tgt_path
            conn = connect(host=impala_host, port=impala_port, auth_mechanism='GSSAPI', kerberos_service_name='impala')
            cur = conn.cursor()
            cur.execute('create role rd_' + team + ';')
            cur.execute('create database if not exists  lhqb_' + team + ';')
            cur.execute('grant all on database lhqb_' + team + ' to  role rd_' + team + ';')

            create_group = "sh adduser.sh -g  gu_hqb_" + team + " rd_" + team
            result = subprocess.call(create_group, shell=True)
            if result != 0:
                return jsonify(res[4])
            else :
                result = mysql_query(
                    "insert into ctl_data.m17_group_role_privilege_info values('gu_hqb_" + team + "','rd_" + team + "','DB','server1','lhqb_" + team + "',null,null,null,'ALL')",
                    ctl_data_conn_conf)
                if isinstance(result, str) and result[:5] == 'error':
                    return result
            create_user = "sh adduser.sh -u U" + user + " Cmb@6666 gr_hqb,gu_hqb_" + team + " Y"
            result = subprocess.call(create_user, shell=True)
            cur.close()
            conn.close()
            if result != 0:
                return jsonify(res[5])
            else:
                result = mysql_query(
                    "replace into ctl_data.m17_team_role_info values('U" + user + "','gu_hqb_" + team + "','lhqb_" + team + "','rd_" + team + "');",
                     ctl_data_conn_conf)
                if isinstance(result, str) and result[:5] == 'error':
                    return result
                return jsonify(res[3])
        else:
            create_user = "sh adduser.sh -u U" + user + " Cmb@6666 gr_hqb,gu_hqb_" + team + " Y"
            result = subprocess.call(create_user, shell=True)
            if result != 0:
                return jsonify(res[5])
            else:
                result = mysql_query(
                    "replace into ctl_data.m17_team_role_info values('U" + user + "','gu_hqb_" + team + "','lhqb_" + team + "','rd_" + team + "');",
                     ctl_data_conn_conf)
                if isinstance(result, str) and result[:5] == 'error':
                    return result
                return jsonify(res[3])
@app.route('/authorize_role/<group>/<role>', methods=['POST', 'GET'])
def authorize_role(group,role):
    subprocess.check_output(hive_kerberos_tgt, shell=True)
    os.environ["KRB5CCNAME"] = hive_kerberos_tgt_path
    conn = connect(host=impala_host, port=impala_port, auth_mechanism='GSSAPI', kerberos_service_name='impala')
    cur = conn.cursor()
    cur.execute("grant role " + role + " to group " + group +";")
    cur.close()
    conn.close()
    return jsonify(res[6])

@app.route('/authorize_table/<table>/<role>/<auth>', methods=['POST', 'GET'])
def authorize_table(table,role,auth):
    subprocess.check_output(hive_kerberos_tgt, shell=True)
    os.environ["KRB5CCNAME"] = hive_kerberos_tgt_path
    conn = connect(host=impala_host, port=impala_port, auth_mechanism='GSSAPI', kerberos_service_name='impala')
    cur = conn.cursor()
    cur.execute("grant " + auth + "  on table " +  table + "  to role " + role  + ";")  #auth  SELECT INSERT REFRESH ALL CREATE(database)
    cur.close()
    conn.close()
    return jsonify(res[6])

@app.route('/authorize_omt/', methods=['POST'])
def authorize_omt():
    result=[]
    apply_tm=str(int(time.time()))
    applyDt=datetime.datetime.now().strftime("%Y-%m-%d")
    data = request.stream.read()
    try:
        datadict=json.loads(data)
        req_typ=datadict['reqTyp']
        req_usr=datadict['reqUsr']
        req_dev_grp=datadict['reqDevGrp']
        is_temp=datadict['isTemp']
        grant_dt=datadict['grantDt']
        revoke_dt=datadict['revokeDt']
        req_dtl=datadict['reqDtl']
    except Exception as err:
        return jsonify({"code":1,"msg":"The json is Error,please check it!"})
        

    if len(req_dtl) == 0:
        return jsonify({"code":1,"msg":"The reqDtl is empty!,please check it!"})
    if is_temp.upper() == "N":
    	  grant_dt=datetime.datetime.now().strftime("%Y-%m-%d")
    	  revoke_dt='9999-12-31'        

    for i,rD in enumerate(req_dtl):
        is_data_area=rD['isDataArea']
        data_area_cd=rD['dataAreaCd']
        obj_nm=rD['objNm']
        if data_area_cd == "":
            return jsonify({"code":1,"msg":"The data_area_cd is empty!,please check it!"})
            
        if is_data_area.upper() == "Y":
            res = mysql_query("insert into ctl_data.m17_grant_privilege_info values('"+req_typ+"','"\
                            +req_usr+"','"+req_dev_grp+"','"+is_data_area+"','"+data_area_cd+"','"+obj_nm+\
                            "','"+is_temp+"','"+grant_dt+"','"+revoke_dt+"','N','N',current_timestamp(),'"+apply_tm+"','" + applyDt + "')",[],ctl_data_conn_conf)
            if isinstance(res, str) and res[:5] == 'error':
                return jsonify({"code":2,"msg":res})
                
        else:
            if obj_nm == "":
                return jsonify({"code":1,"msg":"The obj_nm is empty!,please check it!"})
            else:
                res = mysql_query("insert into ctl_data.m17_grant_privilege_info values('"+req_typ+"','"\
                                +req_usr+"','"+req_dev_grp+"','"+is_data_area+"','"+data_area_cd+"','"+obj_nm+\
                                "','"+is_temp+"','"+grant_dt+"','"+revoke_dt+"','N','N',current_timestamp(),'"+apply_tm+"','" + applyDt + "')",[],ctl_data_conn_conf)
                if isinstance(res, str) and res[:5] == 'error':
                   return  jsonify({"code":2,"msg":res})
    grant_privilege_str="nohup python3 his_grant.py " + apply_tm + " &"
    res_create_group = subprocess.call(grant_privilege_str, shell=True)
    return jsonify({"code":0,"msg":"HDW have got the omt authorization request!"})

@app.route('/change_password/<user>/<curpass>/<newpass>',methods=['POST','GET'])
def change_password(user,curpass,newpass):
    change_password = "sh change_passwd.sh " + user + " " + curpass + " " + newpass 
    result = subprocess.call(change_password,shell=True)
    if result != 0:
        return jsonify(res[8])
    else:
        return jsonify(res[7])

@app.route('/open/dateMigration/hdm/runjob', methods=['POST'])
def run_hdm_job():
    # data = request.stream.read()
    # print('fffff:')
    # print(data)
    # datadict=json.loads(data)
    # datadict = [{'sliceInfo':'20210501,20210503,20210502','taskId':'123456789','srcService':'HDM','srcDatabase':'ACRR_vhis','srcTable':'ACRR_S_RTL_FND_TRX_DER_S','tarDatabase':'ACRR_tdw','tarTable':'arcb_hs_his_bil_bil_dtl_s','incrField':'DW_STAT_DT','startDate':'20100301','endDate':'20200320'}]
    datadict = [{'sliceInfo':'','taskId':'111112','srcService':'HDM','srcDatabase':'ACRR_DATA','srcTable':'JXACRR_S_RTL_FND_TRX_DER_S','tarDatabase':'ACRR_tdw','tarTable':'ACRR_S_RTL_FND_TRX_DER_S','incrField':'DW_STAT_DT','startDate':'20100101','endDate':'20100102'}]
    #datadict = [{'taskId':'123456789','srcService':'HDM','srcDatabase':'ACRR_vhis','srcTable':'ACRR_S_RTL_FND_TRX_DER_Sddd','tarDatabase':'ACRR_tdw','tarTable':'arcb_hs_his_bil_bil_dtl_s','incrField':'DW_STAT_DT','startDate':'20100301','endDate':'20200303'}]
    #datadict = [{'taskId':'123456789','srcService':'HDM','srcDatabase':'ACRR_vhis','srcTable':'NLV62_CI_CONTACT_HISTORY','tarDatabase':'ACRR_tdw','tarTable':'arcb_hs_his_bil_bil_dtl_s','incrField':'DW_STAT_DT','startDate':'20100301','endDate':'20200320'}]
    max_id=mysql_query("select max(id) as maxId from ctl_data.m15_tdw_2_hdw_xtr_dtl_new")
    # max_id=mysql_query_new("select max(id) as maxId from ctl_data.m15_tdw_2_hdw_xtr_dtl_new")
    id=max_id['data'][0]['maxId']
    if id is None:
        id = 1
    else:
        id = id + 1
    num = 0
    dateMoveHdwTaskVOList = []
    ins_sql = "insert into ctl_data.m15_tdw_2_hdw_xtr_dtl_new(id,tdw_db_nm,tdw_tbl_nm,xtr_typ,xtr_col,xtr_date_lst,etl_sts," \
              + "crt_tm,upd_tm,priority,thread_batch,start_date,end_date,task_num) values"
    logging.info('datadict='+str(datadict))
    for item in datadict:
        taskId = item['taskId']
        srcService = item['srcService']
        srcDatabase = item['srcDatabase']
        srcTable=item['srcTable']
        tarDatabase = item['tarDatabase']
        tarTable = item['tarTable']
        startDate = item['startDate']
        endDate = item['endDate']
        sliceInfo = item.get('sliceInfo','').replace('-','')
        if not startDate or not endDate:
            startDate = ''
            endDate = ''
            incrField = ''
            xtrType = 'full'
        else:
            startDate = item['startDate'].replace('-','')
            endDate = item['endDate'].replace('-','')
            incrField = item['incrField']
            xtrType = 'incre'

        dateMoveHdwTask = {}
        dateMoveHdwTask["dmoId"] = taskId
        dateMoveHdwTask["taskNumber"] = taskId
        dateMoveHdwTask["stepMap"] = {}
        dateMoveHdwTaskVOList.append(dateMoveHdwTask)

        batch = num % 3 + 1
        num = num + 1
        tbl_sz = "select tbl_size from ctl_data.m15_hdm_metadata_info where tbl_nm = upper('" + srcTable + "') limit 1"
        result=mysql_query(tbl_sz)
        # result=mysql_query_new(tbl_sz)
        #logging.info('当前表的大小='+str(result['data'][0]['tbl_size']))
        if result['code'] != 0:
            return "{'code':1,'message':'get table size failed','data':{},'success':'false'}"
        if len(result['data']) == 0 or result['data'][0]['tbl_size'] > 600:
            if len(result['data']) == 0:
                split_num = 10
            else:
                split_num = int(result['data'][0]['tbl_size'] / 600.0) + 1
            logging.info('拆分的份数='+str(split_num))
            if len(sliceInfo) == 0:
                days_num = delt_days(startDate,endDate)
                if (days_num + 1) < split_num: 
                    split_num = days_num + 1
                    per_range_days = 1
                    for i in range(split_num):
                        start_dt = add_days(startDate, i*per_range_days)
                        end_dt = add_days(startDate, (i+1)*per_range_days - 1)
                        sql_st = "(" + str(id) + ",'" + srcDatabase + "','" + srcTable + "','" + xtrType + "','" + incrField + "','" + sliceInfo  +  "','I',current_timestamp(),current_timestamp(),20," \
                                 + str(batch) + ",'" + start_dt + "','" + end_dt + "','" + taskId + "')" + ","
                        ins_sql = ins_sql + sql_st
                else:
                    per_range_days = int((days_num+1)/ split_num)
                    for i in range(split_num):
                        start_dt = add_days(startDate, i*per_range_days)
                        if i == split_num - 1:
                            end_dt = endDate
                        else:
                            end_dt = add_days(startDate, (i+1)*per_range_days - 1)
                        sql_st = "(" + str(id) + ",'" + srcDatabase + "','" + srcTable + "','" + xtrType + "','" + incrField + "','" + sliceInfo  +  "','I',current_timestamp(),current_timestamp(),20," \
                                 + str(batch) + ",'" + start_dt + "','" + end_dt + "','" + taskId + "')" + ","
                        ins_sql = ins_sql + sql_st
            else:
                sliceInfoList = sliceInfo.split(',')
                sliceInfoList.sort()
                sliceNums = len(sliceInfoList)
                if sliceNums < split_num:
                    split_num = sliceNums
                    splitSliceNum = 1
                    for i in range(split_num):
                        sliceStartIndex = i*splitSliceNum
                        sliceEndIndex = (i+1)*splitSliceNum
                        sliceInfo = ','.join(sliceInfoList[sliceStartIndex:sliceEndIndex])
                        start_dt = sliceInfoList[sliceStartIndex]
                        end_dt = sliceInfoList[sliceStartIndex]
                        sql_st = "(" + str(id) + ",'" + srcDatabase + "','" + srcTable + "','" + xtrType + "','" + incrField + "','" + sliceInfo  +  "','I',current_timestamp(),current_timestamp(),20," \
                                 + str(batch) + ",'" + start_dt + "','" + end_dt + "','" + taskId + "')" + ","
                        ins_sql = ins_sql + sql_st
                else:
                    splitSliceNum = int(sliceNums/split_num)
                    for i in range(split_num):
                        sliceStartIndex = i*splitSliceNum
                        sliceEndIndex = (i+1)*splitSliceNum
                        if i == split_num - 1:
                            logging.info('最后：')
                            sliceInfo = ','.join(sliceInfoList[sliceStartIndex:])
                            start_dt = sliceInfoList[sliceStartIndex]
                            end_dt = sliceInfoList[sliceNums-1]
                            sql_st = "(" + str(id) + ",'" + srcDatabase + "','" + srcTable + "','" + xtrType + "','" + incrField + "','" + sliceInfo  +  "','I',current_timestamp(),current_timestamp(),20," \
                                     + str(batch) + ",'" + start_dt + "','" + end_dt + "','" + taskId + "')" + ","
                            ins_sql = ins_sql + sql_st
                        else:
                            sliceInfo = ','.join(sliceInfoList[sliceStartIndex:sliceEndIndex])
                            start_dt = sliceInfoList[sliceStartIndex]
                            end_dt = sliceInfoList[sliceEndIndex-1]
                            sql_st = "(" + str(id) + ",'" + srcDatabase + "','" + srcTable + "','" + xtrType + "','" + incrField + "','" + sliceInfo  +  "','I',current_timestamp(),current_timestamp(),20," \
                                     + str(batch) + ",'" + start_dt + "','" + end_dt + "','" + taskId + "')" + ","
                            ins_sql = ins_sql + sql_st
        else:
            logging.info('当前表的大小小于600')
            sql_st = "(" + str(id) + ",'" + srcDatabase + "','" + srcTable + "','incre','" + incrField + "','" + sliceInfo + "','I',current_timestamp(),current_timestamp(),20," + str(batch) + ",'" \
                     + startDate + "','" + endDate + "','" + taskId + "')" + "," 
            ins_sql = ins_sql + sql_st
    logging.info('ins_sql='+ins_sql)           
    result=mysql_query(ins_sql[:-1] + ";")
    # result=mysql_query_new(ins_sql[:-1] + ";")
    if result == 1:
         return "{'code':1,'message':'submit run job failed','data':{},'success':'true'}"
    data_lst = {}
    data_lst["dateMoveHdwTaskVOList"] = dateMoveHdwTaskVOList
    data_lst["sequenceNumber"] = id
    print(data_lst)
    data_lst = json.dumps(data_lst)
    return "{\"code\":0,\"message\":\"success\",\"data\":" + data_lst + ",\"success\":\"false\"}"

@app.route('/open/dateMigration/hdm/task/rerun', methods=['POST'])
def rerun_hdm_job():
    data = request.stream.read()
    datadict=json.loads(data)
    taskid_lst = ""
    for item in datadict:
        taskid_lst = taskid_lst + "'" + item + "',"
    taskid_lst = taskid_lst[:-1]
    upd_sql = "update ctl_data.m15_tdw_2_hdw_xtr_dtl_new set etl_sts='I' where task_num in (" + taskid_lst + ")"
    result = mysql_update(upd_sql,ctl_data_conn_conf)
    return json.dumps(result)

@app.route('/open/dateMigration/hdm/task/cancel', methods=['POST'])
def cancel_hdm_job():
    data = request.stream.read()
    datadict=json.loads(data)
    taskid_lst = ""
    for item in datadict:
        taskid_lst = taskid_lst + "'" + item + "',"
    taskid_lst = taskid_lst[:-1]

    upd_sql = "update ctl_data.m15_tdw_2_hdw_xtr_dtl_new set etl_sts='Q' where task_num in (" + taskid_lst + ")"
    result = mysql_update(upd_sql,ctl_data_conn_conf)
    return json.dumps(result)



@app.route('/open/dateMigration/hdm/datacheck', methods=['POST'])
def data_check():
    result = {}
    kbrs_init('bat_tdd')
    condition = '1=1 '
    data = request.stream.read()
    datadict=json.loads(data)
    sqldict = {}
    print(datadict)
    dbNm = datadict["db_nm"]
    tblNm = datadict["tbl_nm"]
    cheRes = datadict["che_res"]
    starDate = datadict["star_date"]
    endDate = datadict["end_date"]
    size = int(datadict["size"])
    current = int(datadict["current"])
    offsetindex = (current - 1) * size
    if len(dbNm.strip()) != 0 :
       condition += " and upper(db_nm) like '%" + dbNm.upper() + "%'"
    if len(tblNm.strip()) != 0 :
       condition += " and upper(tbl_eng_nm) like '%" + tblNm.upper() + "%'"
    if len(cheRes.strip()) != 0 :
       condition += " and dat_slice_sts = '" + cheRes + "'"
    if len(starDate.strip()) != 0 :
       condition += " and dat_slice_val >= '" + starDate + "'"
    if len(endDate.strip()) != 0 :
       condition += " and dat_slice_val <= '" + endDate + "'"

    #datacheck_sql = "select db_nm,tbl_nm,splt_val,src_service,rcd_num,dat_chk,che_res,mnt_tms "\
    #              + "from ctl_data.slice_che_res_v where " + condition + " order by db_nm,tbl_nm,splt_val limit "\
    #              + str(size) + ' offset ' + str(offsetindex)
    #print(datacheck_sql)
    datacheck_sql = "select db_nm,tbl_eng_nm as tbl_nm,dat_slice_val as splt_val,src_service,dat_slice_cnt as rcd_num,dat_chk,dat_slice_sts as che_res,mnt_tms "\
                   + "from ctl_data.m16_dlm_slice_sts_inf_kudu where " + condition + " order by db_nm,tbl_nm,dat_slice_val limit "\
                   + str(size) + ' offset ' + str(offsetindex)
    print(datacheck_sql)
    #count_sql = "select count(1) from ctl_data.slice_che_res_v where " + condition
    count_sql = "select count(1) from ctl_data.m16_dlm_slice_sts_inf_kudu where " + condition
    datacheckRes = impala_query(datacheck_sql)
    countRes = impala_query(count_sql)
    print(datacheckRes)
    print(countRes)
    if datacheckRes['code'] != 0 or countRes['code'] != 0:
        result['code'] = 1
        result['msg'] = 'Get data check list failed'
        result['data'] = []
        result['total'] = -1
    else:
        result['code'] = 0
        result['msg'] = 'Get data check list successfully'
        result['data'] = datacheckRes['data']
        result['total'] = countRes['data'][0][0]
    return json.dumps(result)

@app.route('/checkUserAndPasswordValid', methods=['POST'])
def checkUserAndPasswordValid():
    data = request.stream.read()
    logging.info('data:'+str(data))
    datadict=json.loads(data)
    username = datadict['username']
    pwd = datadict['pwd']
    cmd = "kinit " + username
    currPwd = pwd + '\n'
    p = subprocess.Popen(cmd, shell = True, stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    p.stdin.write(currPwd)
    output,err = p.communicate()
    logging.info('output:'+str(output))
    logging.info('err:'+str(err))
    result = {}
    if len(err) != 0:
        # return (-1,err)
        #result = "{'code':1,'message':'kinit failed',\"data\":" + err + "}"
        result['code'] = 1
        result['msg'] = str(err)
        result['data'] = []
        return json.dumps(result)
    logging.info('kinit success!')
    # return (0,'kinit success!')
    #result = "{\"code\":0,\"message\":\"kinit success\"}"
    result['code'] = 0
    result['msg'] = 'kinit success'
    result['data'] = []
    return json.dumps(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
    
    #log_file_name =  create_log_file(txdate, folder, file_name)
    #set_log_config()
    logging.info("______________Start rest_server____________")
    logging.info("开始")
    kbrs_init('bat_tdd')
    #hdw_table_create()
    #logging.info("开始22")
    #list = []
    #run_hdm_job()
    #logging.info('返回='+str(cdh_impala_query()))

    # queryHdwTableSliceInfoSql = 'SELECT count(*) FROM pdm_aview.t00_brn_evl_prob_inf_s'
    # queryHdwTableSliceInfoSqlRes = impala_query(queryHdwTableSliceInfoSql)
    # logging.info('queryHdwTableSliceInfoSql='+queryHdwTableSliceInfoSql)
    # logging.info('queryHdwTableSliceInfoSqlRes='+str(queryHdwTableSliceInfoSqlRes))
    # logging.info('queryHdwTableSliceInfoSqlRes1='+str(queryHdwTableSliceInfoSqlRes['data'][0]))
    # logging.info('queryHdwTableSliceInfoSqlRes1='+str(queryHdwTableSliceInfoSqlRes['data'][0][0]))



