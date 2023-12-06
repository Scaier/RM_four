#include "windmill.hpp"

#include <ceres/ceres.h>
#include <cmath>
#include <time.h>
#include <vector>

using namespace std;
using namespace cv;
using namespace ceres;
struct ExponentialResidual
{
    ExponentialResidual(double x, double y) //CostFunction
        : x_(x), y_(y)
    {
    }

    template <typename T>
    bool operator()(const T *const A, const T *const b, const T *const alpha, const T *const omega, T *residual) const
    {
        residual[0] = y_ - A[0] * sin(omega[0] * x_ + alpha[0]) - b[0];
        return true;
    }

private:
    const double x_;
    const double y_;
};

void mySolve(vector<double> x_data, vector<double> y_data, double &A, double &b, double &alpha, double &omega)
{ //拟合
    ceres::Problem problem;
    for (int i = 0; i < x_data.size(); i++)
    {
        ceres::CostFunction *cost_function =
            new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 1, 1, 1, 1>(
                new ExponentialResidual(x_data[i], y_data[i]));
        problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5), &A, &b, &alpha, &omega);
    }
    ceres::Solver::Options options;
    options.logging_type = ceres::SILENT;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
}

int main()
{

    void mySolve(vector<double> x_data, vector<double> y_data, double &A, double &b, double &alpha, double &omega);
    std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    WINDMILL::WindMill wm(t.count());
    cv::Mat image;
    int epoch = 5;
    int maxtime = 500;
    int second = 0;
    double timechange[maxtime / epoch];
    double anglechange[maxtime / epoch];
    double orangle;
    int starttime, endtime, nowtime;
    starttime = t.count();
    while (1)
    {
        t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        image = wm.getMat((double)t.count() / 1000);

        //==========================代码区========================//

        endtime = t.count();

        // 将图像转换为HSV颜色空间
        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

        // 设置红色的阈值范围
        cv::Scalar lower_red(0, 50, 50);
        cv::Scalar upper_red(10, 255, 255);

        // 创建一个掩码，只保留红色区域
        cv::Mat mask;
        cv::inRange(hsv, lower_red, upper_red, mask);

        // 使用cv::threshold函数将红色区域转换为黑色，其他区域转换为白色
        cv::Mat binImg;
        cv::threshold(mask, binImg, 128, 255, cv::THRESH_BINARY);

        // 连通组件标记及统计信息计算
        cv::Mat labels, stats, centroids;
        int numComponents = cv::connectedComponentsWithStats(binImg, labels, stats, centroids);

        cv::Point centroid1, centroid2;

        int m = 0;

        // 绘制每个连通域的边界框
        for (int i = 1; i < numComponents; ++i)
        {
            // 提取当前连通域的mask

            cv::Mat mask = (labels == i);

            int white_pixel_count = countNonZero(mask);

            if (white_pixel_count < 3300)
            {
                // 根据面积过滤连通组件
                int area = stats.at<int>(i, cv::CC_STAT_AREA);
                if (area > 200)
                {
                    // 绘制质心
                    centroid1.x = centroids.at<double>(i, 0);
                    centroid1.y = centroids.at<double>(i, 1);
                    m++;
                }
                if (area <= 200)
                {
                    // 绘制质心
                    centroid2.x = centroids.at<double>(i, 0);
                    centroid2.y = centroids.at<double>(i, 1);
                    m++;
                }
                if (m == 2)
                {
                    break;
                }
            }
        }

        cv::Point center(centroid1.x * 1.2 - centroid2.x * 0.2 , centroid1.y * 1.2 - centroid2.y * 0.2);
        cv::circle(image, center, 10, cv::Scalar(255, 0, 0), 5);

        double k = (double)(centroid2.y - centroid1.y) / (centroid2.x - centroid1.x);

        double angle = atan(k);

        if (centroid2.y < centroid1.y)
        {
            if (centroid2.x < centroid1.x)
            {
                angle = 2 * CV_PI - angle;
            }
            else
            {
                angle = CV_PI - angle;
            }
        }
        else
        {
            if (centroid2.x < centroid1.x)
            {
                angle = -angle;
            }
            else
            {
                angle = CV_PI - angle;
            }
        }
        if (second > 0 && second <= maxtime + 1 && second % epoch == 1)
        {
            if (second == 1)
            {
                nowtime = endtime;
                orangle = angle;
            }
            else
            {
                timechange[(second - 1) / epoch - 1] = (double)(endtime - starttime) / 1000;
                if (angle - orangle < 0)
                {
                    anglechange[(second - 1) / epoch - 1] = (angle - orangle + 2 * CV_PI) / (endtime - nowtime) * 1000;
                }
                else
                {
                    anglechange[(second - 1) / epoch - 1] = (angle - orangle) / (endtime - nowtime) * 1000;
                }
                // cout << anglechange[(second - 1) / epoch - 1] << ' ' << timechange[(second - 1) / epoch - 1] << endl;
                nowtime = endtime;
                orangle = angle;
            }
        }
        if (second > maxtime + 1)
        {
            starttime = t.count();
            vector<double> x_data(timechange, timechange + sizeof(timechange) / sizeof(timechange[0]));
            vector<double> y_data(anglechange, anglechange + sizeof(anglechange) / sizeof(anglechange[0]));

            /* 参数初始化设置，abc初始化为0，噪声 */
            double A = 1;
            double b = 1;
            double omega = 1;
            double alpha = 1;

            mySolve(x_data, y_data, A, b, alpha, omega);
            if (A < 0)
            {
                A = -A;
            }
            if (omega < 0)
            {
                omega = -omega;
            }
            cout << A << " " << b << " " << alpha << " " << omega << endl;

            second = 0;
        }
        imshow("windmill", image);

        second++;

        //=======================================================//

        waitKey(1);
    }

    return 0;
}