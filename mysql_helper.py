#coding=utf-8
from flask import Flask,jsonify,request
import pymysql
import datetime
import logging
import json

app = Flask(__name__)



MYSQL_CONN_CONF = {'user':'root','password':'123456','host':'localhost','database':'paper_exp'}

def make_response(code, message, data=[]):
    return {"code": code, "message": message, "data": data}


def mysql_query(sql_cmd,conn_conf=MYSQL_CONN_CONF):
    result=();
    try:
        conn = pymysql.connect(**conn_conf )
    except pymysql.Error as e:
        logging.error("connect fails!:{}".format(e))
        return make_response(-1, str(e))
        #logging.info("<!--:PRINT{O_RTNCOD:-1,O_RTNMSG:'" + 'connect fails!{}'.format(e) + "'}-->")
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)

    try:
        cursor.execute(sql_cmd)
        result = cursor.fetchall()
    except pymysql.Error as e:
        logging.error("connect fails!:{}".format(e))
        #logging.info("<!--:PRINT{O_RTNCOD:-1,O_RTNMSG:'" + 'connect fails!{}'.format(e) + "'}-->")
        return make_response(-1, str(e))
    finally:
        cursor.close()
        conn.commit()
        conn.close()
    message = "execute '" + sql_cmd + "' successfully"
    return make_response(0, message, result)

def mysql_query_new(sql_cmd,conn_conf=MYSQL_CONN_CONF):
    result=();
    try:
        conn = pymysql.connect(**conn_conf )
    except pymysql.Error as e:
        logging.error("connect fails!:{}".format(e))
        return make_response(-1, e)
        #logging.info("<!--:PRINT{O_RTNCOD:-1,O_RTNMSG:'" + 'connect fails!{}'.format(e) + "'}-->")
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)

    try:
        cursor.execute(sql_cmd)
        result = cursor.fetchall()
    except pymysql.Error as e:
        logging.error("connect fails!:{}".format(e))
        #logging.info("<!--:PRINT{O_RTNCOD:-1,O_RTNMSG:'" + 'connect fails!{}'.format(e) + "'}-->")
        return make_response(-1, e)
    finally:
        cursor.close()
        conn.commit()
        conn.close()
    message = "execute '" + sql_cmd + "' successfully"
    return make_response(0, message, result)


def mysql_update(sql_cmd,conn_conf=MYSQL_CONN_CONF):
    try:
        conn = pymysql.connect(**conn_conf )
    except pymysql.Error as e:
        logging.error("connect fails!:{}".format(e))
        return make_response(-1, e)
        #logging.info("<!--:PRINT{O_RTNCOD:-1,O_RTNMSG:'" + 'connect fails!{}'.format(e) + "'}-->")
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)

    try:
        cursor.execute(sql_cmd)
    except pymysql.Error as e:
        logging.error("connect fails!:{}".format(e))
        return make_response(-1, e)
        #logging.info("<!--:PRINT{O_RTNCOD:-1,O_RTNMSG:'" + 'connect fails!{}'.format(e) + "'}-->")
    finally:
        cursor.close()
        conn.commit()
        conn.close()
    message = "execute '" + sql_cmd + "' successfully"
    return  make_response(0, message)


@app.route('/init_insert/performance/<modelname>', methods=['post','GET'])
def init_insert_performance(modelname):
    sql_cmd="insert into performance(modelname,create_date,test_batch_status) values('" + modelname + "','" + datetime.datetime.now().strftime("%Y-%m-%d") + "','I');"
    return jsonify(mysql_query(sql_cmd))

@app.route('/init_insert/statistic_class/<modelname>/<roc>/<dataset>', methods=['post','GET'])
def init_insert_statistic_class(modelname,roc,dataset):
    sql_cmd="insert into statistic_class(model_name,roc_threshold,create_date,metric_status,dataset) values('" + modelname + "','" + str(roc) + "','" +datetime.datetime.now().strftime("%Y-%m-%d") + "','I','" + dataset +"');"
    res = mysql_query(sql_cmd)
    return jsonify(res)

@app.route('/query/test_batch_status/<modelname>', methods=['post','GET'])
def query_test_batch_status(modelname):
    sql_cmd = "select test_batch_status from performance where modelname = '" + modelname + "';"
    return jsonify(mysql_query(sql_cmd))

@app.route('/query/statistic_status/<modelname>/<roc>/<dataset>', methods=['post','GET'])
def query_statistic_status(modelname,roc,dataset):
    sql_cmd = "select metric_status from statistic_class where model_name = '" + modelname + "' and roc_threshold = '" + str(roc) + "' and dataset = '" + dataset + "';"
    return jsonify(mysql_query(sql_cmd))

@app.route('/update/test_batch_status/<modelname>', methods=['post','GET'])
def update_test_batch_status(modelname):
    sql_cmd = "update performance set test_batch_status='Y' where modelname = '" + modelname + "';"
    return jsonify(mysql_query(sql_cmd))
@app.route('/update/statistic_status/<modelname>/<roc>/<dataset>', methods=['post','GET'])
def update_statistic_status(modelname,roc,dataset):
    sql_cmd = "update statistic_class set metric_status='Y' where model_name = '" + modelname + "' and roc_threshold = '" + str(roc) + "' and dataset = '" + dataset + "';"
    return jsonify(mysql_query(sql_cmd))

@app.route('/update/performance/gflops/<modelname>/<gflops>/<params>', methods=['post','GET'])
def update_gflops(modelname,gflops,params):
    sql_cmd = "update performance set gflops='" + str(gflops) + "',params_M='" + str(params) + "' where modelname = '" + modelname + "';"
    return jsonify(mysql_query(sql_cmd))

@app.route('/update/performance/fps/<modelname>/<fps>', methods=['post','GET'])
def update_fps(modelname,fps):
    sql_cmd = "update performance set fps='" + str(fps) + "' where modelname = '" + modelname + "';"
    return jsonify(mysql_query(sql_cmd))

@app.route('/update/statistic_two_class/', methods=['post','GET'])
def update_two_statistic():
    data = request.stream.read()
    datadict = json.loads(data)
    dataset = datadict['dataset']
    model_name = datadict['modelname']
    roc_threshold = datadict['roc']
    aAcc = datadict['aAcc']
    mioU = datadict['mIoU']
    macc = datadict['mAcc']
    mFscore = datadict['mFscore']
    mPrecision = datadict['mPrecision']
    mRecall = datadict['mRecall']
    mfpr = datadict['mfpr']
    mfnr = datadict['mfnr']
    mkappa = datadict['mkappa']
    mmcc = datadict['mmcc']
    mhloss = datadict['mhloss']
    grmse = datadict.get('Grmse','err')
    gmax = datadict.get('Gmax','err')
    other_iou = datadict['IoU.other']
    water_iou = datadict['IoU.water']
    other_acc = datadict['Acc.other']
    water_acc = datadict['Acc.water']
    other_Fscore = datadict['Fscore.other']
    water_Fscore = datadict['Fscore.water']
    other_Precision = datadict['Precision.other']
    water_Precision = datadict['Precision.water']
    other_Recall = datadict['Recall.other']
    water_Recall = datadict['Recall.water']
    other_fpr = datadict['fpr.other']
    water_fpr = datadict['fpr.water']
    other_fnr = datadict['fnr.other']
    water_fnr = datadict['fnr.water']
    other_kappa = datadict['kappa.other']
    water_kappa = datadict['kappa.water']
    other_mcc = datadict['mcc.other']
    water_mcc = datadict['mcc.water']
    other_hloss = datadict['hloss.other']
    water_hloss = datadict['hloss.water']

    sql_cmd =   "update statistic_class set "    \
                "aAcc='" + str(aAcc) + "', " \
                "mioU='" + str(mioU) + "', " \
                "macc='" + str(macc) + "', " \
                "mFscore='" + str(mFscore) + "', " \
                "mPrecision='" + str(mPrecision) + "', " \
                "mRecall='" + str(mRecall) + "', " \
                "mfpr='" + str(mfpr) + "', " \
                "mfnr='" + str(mfnr) + "', " \
                "mkappa='" + str(mkappa) + "', " \
                "mmcc='" + str(mmcc) + "', " \
                "mhloss='" + str(mhloss) + "', " \
                "grmse='" + str(grmse) + "', " \
                "gmax='" + str(gmax) + "',  " \
                "other_iou='" + str(other_iou) + "', " \
                "water_iou='" + str(water_iou) + "', " \
                "other_acc='" + str(other_acc) + "', " \
                "water_acc='" + str(water_acc) + "', " \
                "other_Fscore='" + str(other_Fscore) + "', " \
                "water_Fscore='" + str(water_Fscore) + "', " \
                "other_Precision='" + str(other_Precision) + "', " \
                "water_Precision='" + str(water_Precision) + "', " \
                "other_Recall='" + str(other_Recall) + "', " \
                "water_Recall='" + str(water_Recall) + "', " \
                "other_fpr='" + str(other_fpr) + "', " \
                "water_fpr='" + str(water_fpr) + "', " \
                "other_fnr='" + str(other_fnr) + "', " \
                "water_fnr='" + str(water_fnr) + "', " \
                "other_kappa='" + str(other_kappa) + "', " \
                "water_kappa='" + str(water_kappa) + "', " \
                "other_mcc='" + str(other_mcc) + "', " \
                "water_mcc='" + str(water_mcc) + "', " \
                "other_hloss='" + str(other_hloss) + "', " \
                "water_hloss='" + str(water_hloss) + "' " \
                " where model_name = '" + model_name + "' and roc_threshold = '" + str(roc_threshold) + "' and dataset = '" + dataset + "';"

    return jsonify(mysql_query(sql_cmd))


@app.route('/update/statistic_three_class/', methods=['post','GET'])
def update_three_statistic():
    data = request.stream.read()
    datadict = json.loads(data)
    dataset = datadict['dataset']
    model_name = datadict['modelname']
    roc_threshold = datadict['roc']
    aAcc = datadict['aAcc']
    mioU = datadict['mIoU']
    macc = datadict['mAcc']
    mFscore = datadict['mFscore']
    mPrecision = datadict['mPrecision']
    mRecall = datadict['mRecall']
    mfpr = datadict['mfpr']
    mfnr = datadict['mfnr']
    mkappa = datadict['mkappa']
    mmcc = datadict['mmcc']
    mhloss = datadict['mhloss']
    grmse = datadict.get('Grmse','err')
    gmax = datadict.get('Gmax','err')
    other_iou = datadict['IoU.other']
    water_iou = datadict['IoU.water']
    sky_iou = datadict['IoU.sky']
    other_acc = datadict['Acc.other']
    water_acc = datadict['Acc.water']
    sky_acc = datadict['Acc.sky']
    other_Fscore = datadict['Fscore.other']
    water_Fscore = datadict['Fscore.water']
    sky_Fscore = datadict['Fscore.sky']
    other_Precision = datadict['Precision.other']
    water_Precision = datadict['Precision.water']
    sky_Precision = datadict['Precision.sky']
    other_Recall = datadict['Recall.other']
    water_Recall = datadict['Recall.water']
    sky_Recall = datadict['Recall.sky']
    other_fpr = datadict['fpr.other']
    water_fpr = datadict['fpr.water']
    sky_fpr = datadict['fpr.sky']
    other_fnr = datadict['fnr.other']
    water_fnr = datadict['fnr.water']
    sky_fnr = datadict['fnr.sky']
    other_kappa = datadict['kappa.other']
    water_kappa = datadict['kappa.water']
    sky_kappa = datadict['kappa.sky']
    other_mcc = datadict['mcc.other']
    water_mcc = datadict['mcc.water']
    sky_mcc = datadict['mcc.sky']
    other_hloss = datadict['hloss.other']
    water_hloss = datadict['hloss.water']
    sky_hloss = datadict['hloss.sky']


    sql_cmd =   "update statistic_class set "    \
                "aAcc='" + str(aAcc) + "', " \
                "mioU='" + str(mioU) + "', " \
                "macc='" + str(macc) + "', " \
                "mFscore='" + str(mFscore) + "', " \
                "mPrecision='" + str(mPrecision) + "', " \
                "mRecall='" + str(mRecall) + "', " \
                "mfpr='" + str(mfpr) + "', " \
                "mfnr='" + str(mfnr) + "', " \
                "mkappa='" + str(mkappa) + "', " \
                "mmcc='" + str(mmcc) + "', " \
                "mhloss='" + str(mhloss) + "', " \
                "grmse='" + str(grmse) + "', " \
                "gmax='" + str(gmax) + "',  " \
                "other_iou='" + str(other_iou) + "', " \
                "water_iou='" + str(water_iou) + "', " \
                "sky_iou='" + str(sky_iou) + "', " \
                "other_acc='" + str(other_acc) + "', " \
                "water_acc='" + str(water_acc) + "', " \
                "sky_acc='" + str(sky_acc) + "', " \
                "other_Fscore='" + str(other_Fscore) + "', " \
                "water_Fscore='" + str(water_Fscore) + "', " \
                "sky_Fscore='" + str(sky_Fscore) + "', " \
                "other_Precision='" + str(other_Precision) + "', " \
                "water_Precision='" + str(water_Precision) + "', " \
                "sky_Precision='" + str(sky_Precision) + "', " \
                "other_Recall='" + str(other_Recall) + "', " \
                "water_Recall='" + str(water_Recall) + "', " \
                "sky_Recall='" + str(sky_Recall) + "', " \
                "other_fpr='" + str(other_fpr) + "', " \
                "water_fpr='" + str(water_fpr) + "', " \
                "sky_fpr='" + str(sky_fpr) + "', " \
                "other_fnr='" + str(other_fnr) + "', " \
                "water_fnr='" + str(water_fnr) + "', " \
                "sky_fnr='" + str(sky_fnr) + "', " \
                "other_kappa='" + str(other_kappa) + "', " \
                "water_kappa='" + str(water_kappa) + "', " \
                "sky_kappa='" + str(sky_kappa) + "', " \
                "other_mcc='" + str(other_mcc) + "', " \
                "water_mcc='" + str(water_mcc) + "', " \
                "sky_mcc='" + str(sky_mcc) + "', " \
                "other_hloss='" + str(other_hloss) + "', " \
                "water_hloss='" + str(water_hloss) + "', " \
                "sky_hloss='" + str(sky_hloss) + "' " \
                " where model_name = '" + model_name + "' and roc_threshold = '" + str(roc_threshold) + "' and dataset = '" + dataset + "';"

    return jsonify(mysql_query(sql_cmd))





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    logging.info("______________Start rest_server____________")
    logging.info("开始")
    


    


